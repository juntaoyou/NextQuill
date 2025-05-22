import torch
import datasets
import numpy as np

from tqdm import tqdm
from utils.templates import Qwen2PromptTemplate


class PersonalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                main_dataset,
                meta_dataset, 
                llm_tokenizer,
                max_length=2048,
                max_his_len=8,
                training=True
    ):
        self.main_dataset = main_dataset
        self.meta_dataset = meta_dataset
        self.max_his_len = max_his_len
        self.llm_tokenizer = llm_tokenizer
        self.total_len = len(self.main_dataset)
        self.cnt = 0
        
        system_prompt = (
            f"Given the title and description of an item, "
            f"along with the user's past reviews "
            f"and the output review rating and review title, "
            f"generate a personalized item review for the user.\n"
        )
        self.pt = Qwen2PromptTemplate(system_prompt)

        self.processed_data = []

        for idx in tqdm(range(self.total_len), desc=f"Pre-Processing data"):
            profile = self.main_dataset[idx]["profile"]
            for p in profile:
                asin = p["asin"]
                p_item_title, p_item_desc = self.meta_dataset[asin]
                p['item_title'] = p_item_title
                p["item_desc"] = p_item_desc
            profile = sorted(profile, key=lambda x: x["timestamp"], reverse=True)[:self.max_his_len]
            data = self.main_dataset[idx]["data"]
            asin = data["asin"]
            item_title, item_desc = self.meta_dataset[asin]
            tmp_inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"[Output Review Rating]: {data['rating']}\n"
                f"[Output Review Title]: {data['title']}\n"
            )
            tmp_inp_str = self.pt.build_prompt(tmp_inp_str)
            tmp_ids = self.llm_tokenizer(tmp_inp_str, add_special_tokens=False)['input_ids']
            tmp_len = len(tmp_ids)
            avail_len = max_length - tmp_len
            past_reviews = ""
            for tmp_prof_len in range(self.max_his_len, 0, -1):
                past_reviews = "".join([
                    f"[Review {i+1}]:\n"
                    f"- [Item Title]: {profile[i]['item_title']}\n"
                    f"- [Item Description]: {profile[i]['item_desc']}\n"
                    f"- [Review Rating]: {profile[i]['rating']}\n"
                    f"- [Review Title]: {profile[i]['title']}\n"
                    f"- [Review Text]: {profile[i]['text']}\n"
                    for i in range(tmp_prof_len)
                ])
                past_reviews = f"[User's Past Reviews]:\n{past_reviews}\n"
                past_ids = self.llm_tokenizer(past_reviews, add_special_tokens=False)['input_ids']
                if len(past_ids) <= avail_len:
                    break
            if tmp_prof_len == 0:
                continue
            self.cnt += tmp_prof_len
            inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"{past_reviews}"
                f"[Output Review Rating]: {data['rating']}\n"
                f"[Output Review Title]: {data['title']}\n"
            )
            inp_str = self.pt.build_prompt(inp_str)
            total_max_length = max_length + 2048 + 1
            out_str = data["text"]
            inputs = self.llm_tokenizer(inp_str,
                                        max_length=max_length,
                                        truncation=True,
                                        add_special_tokens=False)
            targets = self.llm_tokenizer(out_str,
                                        max_length=max_length, 
                                        truncation=True,
                                        add_special_tokens=False)
            data = {
                'inp_str': inp_str,
                'out_str': out_str,
            }
            if training:
                inputs_id = inputs['input_ids'] + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                attention_mask = inputs['attention_mask'] + targets['attention_mask'] + [1]
                labels = [-100] * len(inputs['input_ids']) + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                if len(inputs_id) < total_max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (total_max_length - len(inputs_id)) + inputs_id
                    attention_mask = [0] * (total_max_length - len(attention_mask)) + attention_mask
                    labels = [-100] * (total_max_length - len(labels)) + labels
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
                data['attention_mask'] = np.array(attention_mask, dtype=np.int64)
                data['labels'] = np.array(labels, dtype=np.int64)
            else:
                inputs_id = inputs['input_ids']
                if len(inputs_id) < max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (max_length - len(inputs_id)) + inputs_id
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
            self.processed_data.append(data)

    def __len__(self):
        return self.total_len

    def get_output(self, idx):
        return self.main_dataset[idx]["data"]["text"]
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

    def get_avg_profile_len(self):
        return self.cnt / self.total_len
    
class CFTDataset(torch.utils.data.Dataset):
    def __init__(self, 
                main_dataset,
                meta_dataset, 
                llm_tokenizer,
                max_length=2048,
                max_his_len=8,
                training=True
    ):
        self.main_dataset = main_dataset
        self.meta_dataset = meta_dataset
        self.max_his_len = max_his_len
        self.llm_tokenizer = llm_tokenizer
        self.total_len = len(self.main_dataset)
        self.cnt = 0
        
        system_prompt = (
            f"Given the title and description of an item, "
            f"along with the user's past reviews "
            f"and the output review rating and review title, "
            f"generate a personalized item review for the user.\n"
        )
        self.pt = Qwen2PromptTemplate(system_prompt)

        self.processed_data = []

        for idx in tqdm(range(self.total_len), desc=f"Pre-Processing data"):
            profile = self.main_dataset[idx]["profile"]
            for p in profile:
                asin = p["asin"]
                p_item_title, p_item_desc = self.meta_dataset[asin]
                p['item_title'] = p_item_title
                p["item_desc"] = p_item_desc
            profile = sorted(profile, key=lambda x: x["timestamp"], reverse=True)[:self.max_his_len]
            datas = self.main_dataset[idx]["data"]
            asin = datas["asin"]
            item_title, item_desc = self.meta_dataset[asin]
            tmp_inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"[Output Review Rating]: {datas['rating']}\n"
                f"[Output Review Title]: {datas['title']}\n"
            )
            tmp_inp_str = self.pt.build_prompt(tmp_inp_str)
            tmp_ids = self.llm_tokenizer(tmp_inp_str, add_special_tokens=False)['input_ids']
            tmp_len = len(tmp_ids)
            avail_len = max_length - tmp_len
            past_reviews = ""
            for tmp_prof_len in range(self.max_his_len, 0, -1):
                past_reviews = "".join([
                    f"[Review {i+1}]:\n"
                    f"- [Item Title]: {profile[i]['item_title']}\n"
                    f"- [Item Description]: {profile[i]['item_desc']}\n"
                    f"- [Review Rating]: {profile[i]['rating']}\n"
                    f"- [Review Title]: {profile[i]['title']}\n"
                    f"- [Review Text]: {profile[i]['text']}\n"
                    for i in range(tmp_prof_len)
                ])
                past_reviews = f"[User's Past Reviews]:\n{past_reviews}\n"
                past_ids = self.llm_tokenizer(past_reviews, add_special_tokens=False)['input_ids']
                if len(past_ids) <= avail_len:
                    break
            if tmp_prof_len == 0:
                continue
            self.cnt += tmp_prof_len
            inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"{past_reviews}"
                f"[Output Review Rating]: {datas['rating']}\n"
                f"[Output Review Title]: {datas['title']}\n"
            )
            inp_str = self.pt.build_prompt(inp_str)
            total_max_length = max_length + 2048 + 1
            out_str = datas["text"]
            inputs = self.llm_tokenizer(inp_str,
                                        max_length=max_length,
                                        truncation=True,
                                        add_special_tokens=False)
            targets = self.llm_tokenizer(out_str,
                                        max_length=max_length, 
                                        truncation=True,
                                        add_special_tokens=False)
            data = {
                'inp_str': inp_str,
                'out_str': out_str,
            }
            if training:
                inputs_id = inputs['input_ids'] + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                attention_mask = inputs['attention_mask'] + targets['attention_mask'] + [1]
                labels = [-100] * len(inputs['input_ids']) + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                if len(inputs_id) < total_max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (total_max_length - len(inputs_id)) + inputs_id
                    attention_mask = [0] * (total_max_length - len(attention_mask)) + attention_mask
                    labels = [-100] * (total_max_length - len(labels)) + labels
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
                data['attention_mask'] = np.array(attention_mask, dtype=np.int64)
                data['labels'] = np.array(labels, dtype=np.int64)
            else:
                inputs_id = inputs['input_ids']
                if len(inputs_id) < max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (max_length - len(inputs_id)) + inputs_id
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
            self.processed_data.append(data)
            
            inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"[User's Past Reviews]:\nNone\n"
                f"[Output Review Rating]: {datas['rating']}\n"
                f"[Output Review Title]: {datas['title']}\n"
            )
            inp_str = self.pt.build_prompt(inp_str)
            inputs = self.llm_tokenizer(inp_str,
                                        max_length=max_length,
                                        truncation=True,
                                        add_special_tokens=False)
            targets = self.llm_tokenizer(out_str,
                                        max_length=max_length, 
                                        truncation=True,
                                        add_special_tokens=False)
            # data = {
            #     'inp_str': inp_str,
            #     'out_str': out_str,
            # }
            if training:
                inputs_id = inputs['input_ids'] + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                attention_mask = inputs['attention_mask'] + targets['attention_mask'] + [1]
                if len(inputs_id) < total_max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (total_max_length - len(inputs_id)) + inputs_id
                    attention_mask = [0] * (total_max_length - len(attention_mask)) + attention_mask
                data['input_ids_without_his'] = np.array(inputs_id, dtype=np.int64)
                data['attention_mask_without_his'] = np.array(attention_mask, dtype=np.int64)
            else:
                inputs_id = inputs['input_ids_without_his']
                if len(inputs_id) < max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (max_length - len(inputs_id)) + inputs_id
                data['input_ids_without_his'] = np.array(inputs_id, dtype=np.int64)
            self.processed_data.append(data)

    def __len__(self):
        return self.total_len

    def get_output(self, idx):
        return self.main_dataset[idx]["data"]["text"]
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

    def get_profile_len(self):
        return self.cnt / self.total_len

class CFTDataset2(torch.utils.data.Dataset):
    def __init__(self, 
                main_dataset,
                meta_dataset, 
                llm_tokenizer,
                max_length=2048,
                max_his_len=8,
                training=True,
                # num_retrieve = 5
    ):
        self.main_dataset = main_dataset
        self.meta_dataset = meta_dataset
        self.max_his_len = max_his_len
        self.llm_tokenizer = llm_tokenizer
        self.total_len = len(self.main_dataset)
        self.cnt = 0
        
        system_prompt = (
            f"Given the title and description of an item, "
            f"along with the user's past reviews "
            f"and the output review rating and review title, "
            f"generate a personalized item review for the user.\n"
        )
        self.pt = Qwen2PromptTemplate(system_prompt)

        self.processed_data = []

        for idx in tqdm(range(self.total_len), desc=f"Pre-Processing data"):
            user_id = self.main_dataset[idx]["user_id"]
            profile = self.main_dataset[idx]["profile"]
            for p in profile:
                asin = p["asin"]
                p_item_title, p_item_desc = self.meta_dataset[asin]
                p['item_title'] = p_item_title
                p["item_desc"] = p_item_desc
            # profile = sorted(profile, key=lambda x: x["timestamp"], reverse=True)[:self.max_his_len]
            datas = self.main_dataset[idx]["data"]
            asin = datas["asin"]
            item_title, item_desc = self.meta_dataset[asin]
            tmp_inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"[Output Review Rating]: {datas['rating']}\n"
                f"[Output Review Title]: {datas['title']}\n"
            )
            tmp_inp_str = self.pt.build_prompt(tmp_inp_str)
            tmp_ids = self.llm_tokenizer(tmp_inp_str, add_special_tokens=False)['input_ids']
            tmp_len = len(tmp_ids)
            avail_len = max_length - tmp_len
            past_reviews = ""
            inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"[Output Review Rating]: {datas['rating']}\n"
                f"[Output Review Title]: {datas['title']}\n"
            )
            corpus = [f"{x['item_title']} {x['item_desc']} {x['rating']} {x['title']} {x['text']}" for x in profile]
            selected_profs = use_contriever(max_his_len, corpus, inp_str, profile)
            # my_profile = [data[0] for data in past_reviews_list]
            
            for tmp_prof_len in range(len(selected_profs), -1, -1):
                past_reviews_raw = "".join([
                    f"[Review {i+1}]:\n"
                    f"- [Item Title]: {selected_profs[i]['item_title']}\n"
                    f"- [Item Description]: {selected_profs[i]['item_desc']}\n"
                    f"- [Review Rating]: {selected_profs[i]['rating']}\n"
                    f"- [Review Title]: {selected_profs[i]['title']}\n"
                    f"- [Review Text]: {selected_profs[i]['text']}\n"
                    for i in range(tmp_prof_len)
                ])
                past_reviews = f"[User's Past Reviews]:\n{past_reviews_raw}\n"
                past_ids = self.llm_tokenizer(past_reviews, add_special_tokens=False)['input_ids']
                if len(past_ids) <= avail_len:
                    break
            if tmp_prof_len == 0:
                continue
            
            self.cnt += tmp_prof_len
            inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"{past_reviews}\n"
                f"[Output Review Rating]: {datas['rating']}\n"
                f"[Output Review Title]: {datas['title']}\n"
            )
            inp_str = self.pt.build_prompt(inp_str)
            total_max_length = max_length + 2048 + 1
            out_str = datas["text"]
            inputs = self.llm_tokenizer(inp_str,
                                        max_length=max_length,
                                        truncation=True,
                                        add_special_tokens=False)
            targets = self.llm_tokenizer(out_str,
                                        max_length=max_length, 
                                        truncation=True,
                                        add_special_tokens=False)
            data = {
                'user_id': user_id,
                'inp_str': inp_str,
                'out_str': out_str,
            }
            
            if training:
                inputs_id = inputs['input_ids'] + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                attention_mask = inputs['attention_mask'] + targets['attention_mask'] + [1]
                labels = [-100] * len(inputs['input_ids']) + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                if len(inputs_id) < total_max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (total_max_length - len(inputs_id)) + inputs_id
                    attention_mask = [0] * (total_max_length - len(attention_mask)) + attention_mask
                    labels = [-100] * (total_max_length - len(labels)) + labels
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
                data['attention_mask'] = np.array(attention_mask, dtype=np.int64)
                data['labels'] = np.array(labels, dtype=np.int64)
            else:
                inputs_id = inputs['input_ids']
                if len(inputs_id) < max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (max_length - len(inputs_id)) + inputs_id
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
            self.processed_data.append(data)

    def __len__(self):
        return self.total_len

    def get_output(self, idx):
        return self.main_dataset[idx]["data"]["text"]
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

    def get_profile_len(self):
        return self.cnt / self.total_len
    
    
class CFTDataset3(torch.utils.data.Dataset):
    def __init__(self, 
                main_dataset,
                meta_dataset, 
                llm_tokenizer,
                max_length=2048,
                max_his_len=8,
                training=True
    ):
        self.main_dataset = main_dataset
        self.meta_dataset = meta_dataset
        self.max_his_len = max_his_len
        self.llm_tokenizer = llm_tokenizer
        self.total_len = len(self.main_dataset)
        self.cnt = 0
        
        system_prompt = (
            f"Given the title and description of an item, "
            f"along with the user's past reviews "
            f"and the output review rating and review title, "
            f"generate a personalized item review for the user.\n"
        )
        self.pt = Qwen2PromptTemplate(system_prompt)

        self.processed_data = []

        for idx in tqdm(range(self.total_len), desc=f"Pre-Processing data"):
            profile = self.main_dataset[idx]["profile"]
            for p in profile:
                asin = p["asin"]
                p_item_title, p_item_desc = self.meta_dataset[asin]
                p['item_title'] = p_item_title
                p["item_desc"] = p_item_desc
            # profile = sorted(profile, key=lambda x: x["timestamp"], reverse=True)[:self.max_his_len]
            data = self.main_dataset[idx]["data"]
            asin = data["asin"]
            item_title, item_desc = self.meta_dataset[asin]
            tmp_inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"[Output Review Rating]: {data['rating']}\n"
                f"[Output Review Title]: {data['title']}\n"
            )
            tmp_inp_str = self.pt.build_prompt(tmp_inp_str)
            tmp_ids = self.llm_tokenizer(tmp_inp_str, add_special_tokens=False)['input_ids']
            tmp_len = len(tmp_ids)
            avail_len = max_length - tmp_len
            past_reviews = ""
            for tmp_prof_len in range(self.max_his_len, 0, -1):
                past_reviews_list = []
                past_reviews = "".join([
                    f"[Review {i+1}]:\n"
                    f"- [Item Title]: {profile[i]['item_title']}\n"
                    f"- [Item Description]: {profile[i]['item_desc']}\n"
                    f"- [Review Rating]: {profile[i]['rating']}\n"
                    f"- [Review Title]: {profile[i]['title']}\n"
                    f"- [Review Text]: {profile[i]['text']}\n"
                    for i in range(tmp_prof_len)
                ])
                past_reviews = f"[User's Past Reviews]:\n{past_reviews}\n"
                past_ids = self.llm_tokenizer(past_reviews, add_special_tokens=False)['input_ids']
                if len(past_ids) <= avail_len:
                    break
            if tmp_prof_len == 0:
                continue
            self.cnt += tmp_prof_len
            import transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained("/NAS/yjt/Llama-2-7b-chat-hf")
            model = AutoModelForCausalLM.from_pretrained(
                    "/NAS/yjt/Llama-2-7b-chat-hf",
                    device_map="auto",
                    torch_dtype=torch.float16,
                    load_in_8bit=True,
                    trust_remote_code=True
            )

            tokenizer.pad_token_id = tokenizer.eos_token_id
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype='auto',
                trust_remote_code=True,
                device_map="auto"
            )
            
            summary_list = []
            inp_str = (
                f"[Item Title]: {item_title}\n"
                f"[Item Description]: {item_desc}\n"
                f"{past_reviews}"
                f"[Output Review Rating]: {data['rating']}\n"
                f"[Output Review Title]: {data['title']}\n"
            )
            inp_str = self.pt.build_prompt(inp_str)
            total_max_length = max_length + 2048 + 1
            out_str = data["text"]
            inputs = self.llm_tokenizer(inp_str,
                                        max_length=max_length,
                                        truncation=True,
                                        add_special_tokens=False)
            targets = self.llm_tokenizer(out_str,
                                        max_length=max_length, 
                                        truncation=True,
                                        add_special_tokens=False)
            data = {
                'inp_str': inp_str,
                'out_str': out_str,
            }
            if training:
                inputs_id = inputs['input_ids'] + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                attention_mask = inputs['attention_mask'] + targets['attention_mask'] + [1]
                labels = [-100] * len(inputs['input_ids']) + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
                if len(inputs_id) < total_max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (total_max_length - len(inputs_id)) + inputs_id
                    attention_mask = [0] * (total_max_length - len(attention_mask)) + attention_mask
                    labels = [-100] * (total_max_length - len(labels)) + labels
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
                data['attention_mask'] = np.array(attention_mask, dtype=np.int64)
                data['labels'] = np.array(labels, dtype=np.int64)
            else:
                inputs_id = inputs['input_ids']
                if len(inputs_id) < max_length:
                    inputs_id = [self.llm_tokenizer.pad_token_id] * (max_length - len(inputs_id)) + inputs_id
                data['input_ids'] = np.array(inputs_id, dtype=np.int64)
            self.processed_data.append(data)

    def __len__(self):
        return self.total_len

    def get_output(self, idx):
        return self.main_dataset[idx]["data"]["text"]
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

    def get_profile_len(self):
        return self.cnt / self.total_len
    
def convert_to_dataset(dataset):
    def gen():
        for data in dataset:
            yield data
    return datasets.Dataset.from_generator(gen)
