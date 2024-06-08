import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# Load the pre-trained IndicBERT model and tokenizer
model_name = 'ai4bharat/indic-bert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Load and prepare the dataset
def load_and_prepare_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    dataset = Dataset.from_dict({'text': lines})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Fine-tune the model
def fine_tune_model(train_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    # Load and prepare the dataset
    dataset_file = '/home/ubuntu/final_cleaned_spb_texts.txt'
    train_dataset = load_and_prepare_dataset(dataset_file)

    # Fine-tune the model
    output_dir = './fine_tuned_indicbert'
    fine_tune_model(train_dataset, output_dir)

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
