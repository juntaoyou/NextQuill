import os
import sys
import json
import torch
import argparse
import evaluate
import warnings
import torch.distributed as dist
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from datasets import load_from_disk
from transformers import set_seed, AutoTokenizer,  AutoModelForCausalLM
from transformers import GenerationConfig
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm
# from peft import PeftModel

from utils.utils import write_to_csv

warnings.filterwarnings("ignore")

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--category", required=True, 
                    choices=["Movies_and_TV", "CDs_and_Vinyl", "Books"])

args = parser.parse_args()
category = args.category

personal_dataset = load_from_disk(f"./data/data/dataset_test_{category}")

model_dir = "output"
subfolders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
model_dirs = [os.path.join(model_dir, f) for f in subfolders]
model_dirs = sorted(model_dirs, key=lambda x: int(x.split("-")[-1]))

best_idx = 0
best_score = 0
for i in range(5):
    result_path = f"./output/result_{i}.json"
    if not os.path.exists(result_path):
        continue
    with open(result_path, "r") as f:
        result = json.load(f)
    if result["meteor"] > best_score:
        best_score = result["meteor"]
        best_idx = i
best_idx = -1
model_dir = model_dirs[best_idx]

model_dir = "output"
llm_tokenizer = AutoTokenizer.from_pretrained(model_dir)
personal_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# personal_model = PeftModel.from_pretrained(personal_model, lora_path, torch_dtype=torch.bfloat16)
personal_model.resize_token_embeddings(len(llm_tokenizer))
personal_model.eval()

personal_model.generation_config = GenerationConfig(
    temperature=0.8,           
    top_p=0.95,                  
    repetition_penalty=1.2,   
    max_new_tokens=2048,
    eos_token_id=llm_tokenizer.eos_token_id,
    pad_token_id=llm_tokenizer.pad_token_id,
)

references = personal_dataset["out_str"]
predictions = []

dataloader = DataLoader(
    personal_dataset,
    batch_size=200,
    shuffle=False,
    collate_fn=default_data_collator
)

for sample in tqdm(dataloader, desc="Generating data"):
    batch = {k: v.to(personal_model.device) for k, v in sample.items()}
    with torch.no_grad():
        generated_ids = personal_model.generate(
            input_ids=batch["input_ids"]
        )
    generated_ids = generated_ids[:, len(batch["input_ids"][0]):]
    texts = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    predictions.extend(texts)

bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load('rouge')
meteor_metric = evaluate.load('meteor')
result_bleu = bleu_metric.compute(predictions=predictions,
                                references=references)
result_rouge = rouge_metric.compute(predictions=predictions,
                                    references=references)
result_meteor = meteor_metric.compute(predictions=predictions,
                                    references=references)
result = {
    "model": model_dir,
    "rouge-1": result_rouge["rouge1"],
    "rouge-L": result_rouge["rougeL"],
    "meteor": result_meteor['meteor'],
    "bleu": result_bleu["score"],
}
print(result)

write_to_csv(f"demo-rag-{category}", "rouge-1", result["rouge-1"],file_path="../result
.csv")
write_to_csv(f"demo-rag-{category}", "rouge-L", result["rouge-L"],file_path="../result.csv")
write_to_csv(f"demo-rag-{category}", "meteor", result["meteor"],file_path="../result.csv")
write_to_csv(f"demo-rag-{category}", "bleu", result["bleu"],file_path="../result.csv")
