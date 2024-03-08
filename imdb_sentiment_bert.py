from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load the IMDB dataset
dataset = load_dataset('imdb')

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Preprocess the data
def preprocess_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

tokenized_imdb = dataset.map(preprocess_function, batched=True)
# encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Define the training arguments
training_args = TrainingArguments(
    output_dir="imdb_bert_classifier",
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

import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    # data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate the model
trainer.evaluate()