import os
import sys
import torch
import warnings
import torch.distributed as dist
os.environ["HF_DATASETS_CACHE"] = ''
os.environ['HF_HOME'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer,TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import set_seed, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import torch.nn.functional as F
warnings.filterwarnings("ignore")
import logging
import deepspeed
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging_file = "LOG_FILE"
file_handler = logging.FileHandler(logging_file)
file_handler.setLevel(logging.DEBUG)


formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)


class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

set_seed(42)
world_size = int(os.environ.get("WORLD_SIZE", 1))
device_map = "auto"
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

class CustomTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"output/checkpoint-{self.state.global_step}"
        if not os.path.exists(checkpoint_folder):
            os.mkdir(checkpoint_folder)
        self.save_model(checkpoint_folder)
        self.tokenizer.save_pretrained(checkpoint_folder)
        print(f"Checkpoint saved to {checkpoint_folder}")

class CustomTrainer2(Trainer):
    def __init__(self, alpha = 0.01, weight_strategy = "JS", causal = False, rag = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_strategy = weight_strategy
        self.alpha = alpha
        self.causal = causal
        self.have_print = True
        self.rag = rag
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if not model.training:
            outputs = model(**inputs)
            loss = outputs['loss']
            return (loss, outputs) if return_outputs else loss
        
            
        inputs0 = {"input_ids": inputs['input_ids'], "attention_mask":inputs["attention_mask"],"labels": inputs['labels']}
        inputs1 = {"input_ids": inputs['input_ids_without_his'], "attention_mask":inputs["attention_mask_without_his"],"labels": inputs['labels']}
        if not self.rag:
            outputs = model(**inputs1)
            loss = outputs['loss']
            return (loss, outputs) if return_outputs else loss
            
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs0 = {**inputs0, **loss_kwargs}
        outputs0 = model(**inputs0)
        loss = outputs0['loss']
        outputs1 = model(**inputs1)
        logits0 = outputs0['logits']
        logits1 = outputs1['logits']
        del outputs1
        shift_logits = logits0[..., :-1, :].contiguous()
        vocab_size = shift_logits.shape[-1]
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = inputs0['labels'][..., 1:].contiguous().to(shift_logits.device)
        shift_labels = shift_labels.view(-1)
        shift_logits1 = logits1[..., :-1, :].contiguous().to(shift_logits.device)
        shift_logits1 = shift_logits1.view(-1, vocab_size)
        del logits0, logits1
        if model.training:
            if not self.causal:
                input = F.softmax(shift_logits, dim = -1)
                target = F.softmax(shift_logits1, dim = -1)
                del shift_logits1
                if self.weight_strategy == 'JS':
                    mid = torch.log((input + target) / 2)
                    KLdiv = torch.nn.KLDivLoss(reduction = 'none')
                    weight = KLdiv(mid, input) + KLdiv(mid, target)
                    weight = torch.abs(torch.sum(weight, dim = -1))
                elif self.weight_strategy == 'Target':
                    index0 = [i for i in range(shift_labels.shape[0])]
                    _p1 = input[index0, shift_labels]
                    _p2 = target[index0, shift_labels]
                    _flag = shift_labels > -100
                    p1 = torch.where(_flag, _p1, torch.zeros_like(_p1))
                    p2 = torch.where(_flag, _p2, torch.zeros_like(_p2))
                    threshold = 0.05
                    _compare = p1 - p2 > threshold
                    compare = _compare.float().to(shift_labels.device)
                    compare_sum = torch.sum(compare, dim=-1)
                    print("Ratio:", compare_sum / compare.shape[0])
                    high_weight, low_weight = 0.9, 0.1
                    weight = torch.where(_compare, high_weight*torch.ones_like(_compare), low_weight*torch.ones_like(_compare))

                else:
                    raise ValueError("Invalid Weighting Strategy!!!")
                _flag = shift_labels > -100
                weight = torch.where(_flag,weight,torch.zeros_like(weight)).to(shift_logits.device)
                s = weight.sum().to(shift_logits.device)
                weight = weight / s
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits, shift_labels)
                loss = (weight * loss).sum()
                del shift_logits, weight
                torch.cuda.empty_cache()
            else:
                if self.have_print:
                    print("Alpha:", self.alpha)
                    self.have_print = False
                shift_diff_logits = shift_logits - shift_logits1
                del shift_logits1
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)
                loss_diff = loss_fct(shift_diff_logits, shift_labels)
                loss += self.alpha * loss_diff   
                del shift_logits, shift_diff_logits
                torch.cuda.empty_cache()

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs0) if return_outputs else loss
    

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
llm_model_name = "Qwen/Qwen2.5-7B"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_tokenizer.padding_side = "left"

personal_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    device_map=device_map,
    # quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
personal_model.gradient_checkpointing_enable()
# personal_model = prepare_model_for_kbit_training(personal_model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # , "k_proj", "out_proj"
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
personal_model = get_peft_model(personal_model, peft_config=peft_config)

device_num = torch.cuda.device_count()
# personal_model = get_peft_model(personal_model, peft_config = peft_config)
print(personal_model)
print_trainable_parameters(personal_model)
if not ddp and torch.cuda.device_count() > 1:
        personal_model.is_parallelizable = True
        personal_model.model_parallel = True 

training_args = TrainingArguments(
    num_train_epochs=5,
    output_dir=f"output",
    logging_steps=10,
    save_strategy="epoch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    optim="adamw_torch",
    learning_rate=5e-6,
    weight_decay=0.025,
    warmup_ratio=0.01,
    bf16=True,
    deepspeed="./deepspeed/ds_z1_config.json",
    report_to="none",
    remove_unused_columns=False,
    ddp_find_unused_parameters=False if ddp else None,
    run_name="demo-rag",
    # load_best_model_at_end=True,
)

personal_dataset = load_from_disk(f"./data/data/dataset_train4").remove_columns(["inp_str", "out_str"])
# val_dataset = load_from_disk("./data/data/dataset_val2").remove_columns(["inp_str", "out_str"])
trainer = CustomTrainer2(
# trainer = Trainer(
    model=personal_model,
    args=training_args,
    alpha = 0.1,
    causal = True, 
    # rag = False,
    train_dataset=personal_dataset,
    # eval_dataset = val_dataset,
    tokenizer=llm_tokenizer,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=1)],
)
personal_model.config.use_cache=False

print("train start")
trainer.train()
print("train done")
if dist.is_initialized():
    dist.destroy_process_group()
sys.exit(0)
