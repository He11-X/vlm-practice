import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "./experiment_checkpoints"
image_path = sys.argv[2] if len(sys.argv) > 2 else None
question = sys.argv[3] if len(sys.argv) > 3 else "Что изображено на картинке?"

if image_path is None:
    print("Usage: python inference.py <model_path> <image_path> <question>")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlavaForConditionalGeneration.from_pretrained(model_path).to(device)
processor = LlavaProcessor.from_pretrained(model_path)

image = Image.open(image_path).convert("RGB")
prompt = f"USER: <image>\n{question}\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)
answer = processor.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
print(answer)
