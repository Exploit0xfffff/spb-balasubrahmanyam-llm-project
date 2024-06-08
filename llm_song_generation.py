import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import requests
from bs4 import BeautifulSoup
import argparse

# Function to extract text from a URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract text from specific HTML elements that are likely to contain lyrics
    lyrics_div = soup.find('div', class_='lyrics')
    if lyrics_div:
        text = lyrics_div.get_text()
    else:
        # Attempt to find other common elements that might contain lyrics
        lyrics_containers = soup.find_all(['div', 'span'], class_=['lyric-text', 'song-lyrics', 'lyrics-container', 'lyrics', 'text'])
        if lyrics_containers:
            text = ' '.join([container.get_text() for container in lyrics_containers])
        else:
            # Further refine the search to target more specific elements
            lyrics_paragraphs = soup.find_all('p', class_='verse')
            if lyrics_paragraphs:
                text = ' '.join([p.get_text() for p in lyrics_paragraphs])
            else:
                # Fallback to extracting text from all paragraphs
                paragraphs = soup.find_all('p')
                text = ' '.join([p.get_text() for p in paragraphs])
    return text

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

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a song in the style of SP Balasubrahmanyam.")
    parser.add_argument("prompt", type=str, help="The prompt to generate the song.")
    parser.add_argument("--language", type=str, default='en', help="The language of the song (default: 'en').")
    args = parser.parse_args()

    generated_song = generate_song(args.prompt, language=args.language)
    print(generated_song)
