import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from datasets import load_dataset
import config
from tqdm import tqdm
import random
import re

random.seed(config.SEED)
torch.manual_seed(config.SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlavaForConditionalGeneration.from_pretrained(config.OUTPUT_DIR).to(device)
processor = LlavaProcessor.from_pretrained(config.OUTPUT_DIR)

dataset = load_dataset("deepvk/MMBench-ru", split=config.MMBENCH_SPLIT)
if config.EVAL_SAMPLES > 0 and config.EVAL_SAMPLES < len(dataset):
    indices = random.sample(range(len(dataset)), config.EVAL_SAMPLES)
    dataset = dataset.select(indices)

def extract_answer_letter(text):
    text = text.strip()
    match = re.search(r'^(?:[А-ЯA-Z]?\s*[\(\)]?\s*)?([A-Ha-h])', text)
    if match:
        return match.group(1).upper()
    match = re.search(r'ответ[:\s]+([A-Ha-h])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

correct = 0
total = 0
for example in tqdm(dataset):
    image = example["image"]
    question = example["question"]
    options = example["options"]
    answer_idx = example["answer"]
    options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    prompt = f"USER: <image>\n{question}\n{options_text}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
    generated = processor.decode(outputs[0], skip_special_tokens=True)
    pred = generated.split("ASSISTANT:")[-1].strip()
    pred_letter = extract_answer_letter(pred)
    if pred_letter:
        pred_idx = ord(pred_letter) - 65
        if pred_idx == answer_idx:
            correct += 1
    total += 1

accuracy = correct / total
print(f"MMBench-ru accuracy: {accuracy:.4f} (on {total} samples)")
with open("mmbench_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\nSamples: {total}\n")
