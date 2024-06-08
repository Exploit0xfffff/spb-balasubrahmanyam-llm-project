import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset

# Load the tokenizer and model from the local files
tokenizer = AutoTokenizer.from_pretrained("./albert-indic64k", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")

# Load the model weights from the checkpoint file
checkpoint = torch.load("./separate_script_indicbart_model.ckpt", map_location=torch.device('cpu'))
model.resize_token_embeddings(64015)  # Adjust the model's vocabulary size to match the checkpoint
model.load_state_dict(checkpoint, strict=False)

# Load the dataset
with open('telugu_lyrics_dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Convert the dataset to the format expected by the model
dataset = Dataset.from_dict(data)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['lyrics'], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
