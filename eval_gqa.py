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

test_dataset = load_dataset("deepvk/GQA-ru", split=config.GQA_TEST_SPLIT)
if config.EVAL_SAMPLES > 0 and config.EVAL_SAMPLES < len(test_dataset):
    indices = random.sample(range(len(test_dataset)), config.EVAL_SAMPLES)
    test_dataset = test_dataset.select(indices)

def normalize(s):
    return re.sub(r'[^\w\s]', '', s).lower().strip()

correct = 0
total = 0
for example in tqdm(test_dataset):
    image = example["image"]
    question = example["question"]
    true_answer = example["answer"]
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    pred = generated_text.split("ASSISTANT:")[-1].strip()
    if normalize(pred) == normalize(true_answer):
        correct += 1
    total += 1

accuracy = correct / total
print(f"GQA-ru accuracy: {accuracy:.4f} (on {total} samples)")
with open("gqa_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\nSamples: {total}\n")
