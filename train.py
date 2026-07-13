import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor, Trainer, TrainingArguments
from data_utils import prepare_gqa_dataset, collate_fn
import config
import os
import json

torch.manual_seed(config.SEED)

processor = LlavaProcessor.from_pretrained(config.PROCESSOR_NAME)
model = LlavaForConditionalGeneration.from_pretrained(
    config.MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

train_full = prepare_gqa_dataset(config.GQA_TRAIN_SPLIT, processor, config.MAX_LENGTH)
train_size = int(0.9 * len(train_full))
val_size = len(train_full) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_full, [train_size, val_size],
                                                           generator=torch.Generator().manual_seed(config.SEED))

training_args = TrainingArguments(
    output_dir=config.OUTPUT_DIR,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    gradient_accumulation_steps=config.GRAD_ACCUM,
    num_train_epochs=config.EPOCHS,
    learning_rate=config.LEARNING_RATE,
    warmup_ratio=config.WARMUP_RATIO,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    dataloader_num_workers=4,
    remove_unused_columns=False,
    seed=config.SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

trainer.train()
trainer.save_model(config.OUTPUT_DIR)
processor.save_pretrained(config.OUTPUT_DIR)

with open(os.path.join(config.OUTPUT_DIR, "experiment_config.json"), "w") as f:
    json.dump({k: v for k, v in vars(config).items() if not k.startswith("__")}, f, indent=2)
