
from datasets import load_dataset

imdb = load_dataset("imdb")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, GPT2Config
import torch

config = GPT2Config.from_pretrained('gpt2')
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2", num_labels=2, pad_token_id = tokenizer.pad_token_id, id2label=id2label, label2id=label2id
)

# For GPT2 using bfloat16 results in worse eval accuracy on imdb sentiment analysis
# model = AutoModelForSequenceClassification.from_pretrained(
#     "gpt2", num_labels=2, pad_token_id = tokenizer.pad_token_id, id2label=id2label, label2id=label2id, torch_dtype=torch.bfloat16
# )
model.resize_token_embeddings(len(tokenizer))
max_seq_len = model.config.n_positions
hidden_size = model.config.n_embd
torch_dtype = model.config.torch_dtype

def preprocess_function(examples):
    # return tokenizer(examples["text"], truncation=True, padding=True, max_length=max_seq_len-1)
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="imdb_gpt2_classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    use_cpu=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    # data_collator=data_collator,
    compute_metrics=compute_metrics,
)

from optimum.bettertransformer import BetterTransformer

# FlashAttention-2 can only be used when the model’s dtype is fp16 or bf16. Make sure to cast your model to the appropriate dtype and load them on a supported device before using FlashAttention-2.
if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
    model = BetterTransformer.transform(model)
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        trainer.train()
else: 
    trainer.train()

