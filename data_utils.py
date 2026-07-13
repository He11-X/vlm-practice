import torch
from datasets import load_dataset
from transformers import LlavaProcessor
import config

def load_gqa(split):
    return load_dataset("deepvk/GQA-ru", split=split)

def load_mmbench(split="dev"):
    return load_dataset("deepvk/MMBench-ru", split=split)

def load_llava_instruct():
    return load_dataset("deepvk/LLaVA-Instruct-ru")

def prepare_gqa_dataset(split, processor, max_length=2048, seed=config.SEED):
    dataset = load_gqa(split)
    dataset = dataset.shuffle(seed=seed)
    def process(example):
        image = example["image"]
        question = example["question"]
        answer = example["answer"]
        text = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        inputs = processor(text=text, images=image, return_tensors="pt",
                           padding="max_length", truncation=True, max_length=max_length)
        pixel_values = inputs.pixel_values.squeeze(0)
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        prompt_part = f"USER: <image>\n{question}\nASSISTANT: "
        prompt_tokens = processor.tokenizer(prompt_part, add_special_tokens=False)['input_ids']
        labels = torch.full_like(input_ids, -100)
        answer_tokens = processor.tokenizer(answer, add_special_tokens=False)['input_ids']
        prompt_len = len(prompt_tokens)
        real_len = attention_mask.sum().item()
        answer_start = prompt_len
        answer_end = real_len
        if answer_end > max_length:
            answer_end = max_length
        labels[answer_start:answer_end] = input_ids[answer_start:answer_end]
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    return dataset.map(process, remove_columns=dataset.column_names)

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
