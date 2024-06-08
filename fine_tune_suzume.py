import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# Load the pre-trained BLOOM-560m model and tokenizer
model_name = 'bigscience/bloom-560m'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

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
        per_device_train_batch_size=1,  # Further reduced batch size to mitigate memory issues
        gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_loss=custom_compute_loss  # Use custom loss function
    )
    trainer.train()

# Custom loss function
def custom_compute_loss(model, inputs, return_outputs=False):
    outputs = model(**inputs)
    logits = outputs.get("logits")
    labels = inputs.get("labels")
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    # Load and prepare the dataset
    dataset_file = '/home/ubuntu/final_cleaned_spb_texts.txt'
    train_dataset = load_and_prepare_dataset(dataset_file)

    # Fine-tune the model
    output_dir = './fine_tuned_bloom'
    fine_tune_model(train_dataset, output_dir)

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
