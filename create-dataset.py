import os
import sys
os.environ["HF_DATASETS_CACHE"] = '/NAS/yjt/hf_dataset2'
os.environ['HF_HOME'] = '/NAS/yjt/datasets'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

from transformers import AutoTokenizer, set_seed
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from personal_dataset import PersonalDataset, convert_to_dataset, CFTDataset, CFTDataset2

set_seed(42)

categories = ["Books","Movies_and_TV", "CDs_and_Vinyl"]

# train_dataset1 = load_from_disk("/NAS/yjt/demo-rag/data/dataset/dataset_train_Books")
# train_dataset2 = load_from_disk("/NAS/yjt/demo-rag/data/dataset/dataset_train_CDs_and_Vinyl")
# train_dataset3 = load_from_disk("/NAS/yjt/demo-rag/data/dataset/dataset_train_Movies_and_TV")
# train_dataset = concatenate_datasets([train_dataset1, train_dataset2, train_dataset3])
# # train_dataset.save_to_disk("./data/dataset_train_cft")

# val_dataset1 = load_from_disk("/NAS/yjt/demo-rag/data/dataset/dataset_test_Books")
# val_dataset2 = load_from_disk("/NAS/yjt/demo-rag/data/dataset/dataset_test_CDs_and_Vinyl")
# val_dataset3 = load_from_disk("/NAS/yjt/demo-rag/data/dataset/dataset_test_Movies_and_TV")
# val_dataset = concatenate_datasets([val_dataset1, val_dataset2, val_dataset3])
# val_dataset.save_to_disk("./data/dataset_test_with_user_ids")

meta_dataset1 = load_from_disk("/NAS/yjt/demo-rag/data/dataset/dataset_meta_Books")
meta_dataset2 = load_from_disk("/NAS/yjt/demo-rag/data/dataset/dataset_meta_CDs_and_Vinyl")
meta_dataset3 = load_from_disk("/NAS/yjt/demo-rag/data/dataset/dataset_meta_Movies_and_TV")
meta_dataset = concatenate_datasets([meta_dataset1, meta_dataset2, meta_dataset3])

meta_dataset = dict(zip(meta_dataset["asin"],
                        zip(meta_dataset["title"],
                            meta_dataset["description"])))

llm_model_name = "/NAS/yjt/Qwen2.5-3B"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_tokenizer.padding_side = "left"

# personal_dataset = PersonalDataset(val_dataset,
#                         meta_dataset,
#                         llm_tokenizer=llm_tokenizer,
#                         training = False
#                         )
# print(personal_dataset.get_avg_profile_len())
# hf_dataset = convert_to_dataset(personal_dataset)
# hf_dataset.save_to_disk("./data/dataset_train")

# personal_dataset = PersonalDataset(val_dataset,
#                         meta_dataset,
#                         llm_tokenizer=llm_tokenizer)
# hf_dataset = convert_to_dataset(personal_dataset)
# hf_dataset.save_to_disk("./data/dataset_val")

for i, category in enumerate(categories):
    test_main_dataset = load_from_disk(f"/NAS/yjt/demo-rag/data/dataset/dataset_test_{category}")
    test_dataset = CFTDataset(
        test_main_dataset, 
        meta_dataset,
        llm_tokenizer=llm_tokenizer,
        # training=False
    )
    hf_dataset = convert_to_dataset(test_dataset)
    hf_dataset.save_to_disk(f"./data/dataset_test2_{category}")

# personal_dataset =  CFTDataset(
#     val_dataset,
#     meta_dataset,
#     llm_tokenizer=llm_tokenizer,
#     training=False
# )
# hf_dataset = convert_to_dataset(personal_dataset)
# hf_dataset.save_to_disk("./data/dataset_test")

# for i, category in enumerate(categories):
#     test_main_dataset = load_dataset(
#         "SnowCharmQ/DPL-main",
#         category,
#         split="test"
#     ).map(lambda _: {"category": category})
#     test_dataset = PersonalDataset(
#         test_main_dataset, 
#         meta_dataset,
#         llm_tokenizer=llm_tokenizer,
#         training=False
#     )
#     hf_dataset = convert_to_dataset(test_dataset)
#     hf_dataset.save_to_disk(f"data/dataset_test_{category}")
