import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import requests
import argparse

# Function to extract text from a URL

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate songs
def generate_song(prompt, language='en', max_length=300):  # Increased max_length to 300
    print(f"Generating song with prompt: {prompt} in language: {language}")

    # Adjust model and tokenizer based on language
    if language == 'te':
        model_name = 'ai4bharat/indic-gpt-te'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif language == 'hi':
        model_name = 'ai4bharat/indic-gpt-hi'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif language == 'mr':
        model_name = 'ai4bharat/indic-gpt-mr'
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        model_name = 'bigscience/bloom-560m'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    print(f"Encoded inputs: {inputs}")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        attention_mask=torch.ones(inputs.shape, dtype=torch.long),  # Set attention mask
        pad_token_id=tokenizer.eos_token_id,  # Set pad token id
        temperature=0.7,  # Control the creativity of the output
        top_k=50,  # Limit the sampling pool to top k tokens
        top_p=0.95,  # Nucleus sampling
        do_sample=True  # Enable sample-based generation
    )
    print(f"Generated outputs: {outputs}")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to pre-train the model
def pretrain_model(dataset_path, model_name='gpt2', output_dir='./model_output', epochs=3, batch_size=4):
    print(f"Pre-training model with dataset: {dataset_path}")

    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Load the dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=128
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a song in the style of SP Balasubrahmanyam or pre-train the model.")
    parser.add_argument("action", type=str, choices=["generate", "pretrain"], help="The action to perform: 'generate' or 'pretrain'.")
    parser.add_argument("prompt_or_dataset", type=str, help="The prompt to generate the song or the path to the dataset for pre-training.")
    parser.add_argument("--language", type=str, default='en', help="The language of the song (default: 'en').")
    parser.add_argument("--model_name", type=str, default='gpt2', help="The model name for pre-training (default: 'gpt2').")
    parser.add_argument("--output_dir", type=str, default='./model_output', help="The output directory for the pre-trained model (default: './model_output').")
    parser.add_argument("--epochs", type=int, default=3, help="The number of epochs for pre-training (default: 3).")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size for pre-training (default: 4).")
    args = parser.parse_args()

    if args.action == "generate":
        generated_song = generate_song(args.prompt_or_dataset, language=args.language)
        print(generated_song)
    elif args.action == "pretrain":
        pretrain_model(args.prompt_or_dataset, model_name=args.model_name, output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size)
