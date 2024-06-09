#!/usr/bin/python3

import sys
import os
import random

os.environ['PYTHONPATH'] = '/home/ubuntu/.local/lib/python3.10/site-packages'

print("sys.executable:", sys.executable)
print("sys.path:", sys.path)
print("Environment Variables:", os.environ)
print("PYTHONPATH:", os.environ.get('PYTHONPATH', 'Not Set'))
print("sys.path at runtime:", sys.path)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
import torch

# Load the tokenizer and model from the local files
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")

# Load the model weights from the pytorch_model.bin file
model_weights_path = "./model_output/pytorch_model.bin"
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')), strict=False)

# Define a list of prompts for generating songs in Telugu
prompts = [
    "ఈ పాట గురించి ప్రేమ, బాధ, ఆనందం, మరియు జీవితం గురించి ఒక పూర్తి పాట రాయండి. పాట ప్రారంభం:",
    "ఈ పాట గురించి స్నేహం, ఆశ, మరియు కలలు గురించి ఒక పూర్తి పాట రాయండి. పాట ప్రారంభం:",
    "ఈ పాట గురించి కుటుంబం, ఆనందం, మరియు ఆశయం గురించి ఒక పూర్తి పాట రాయండి. పాట ప్రారంభం:",
    "ఈ పాట గురించి ప్రకృతి, సౌందర్యం, మరియు ప్రశాంతత గురించి ఒక పూర్తి పాట రాయండి. పాట ప్రారంభం:",
    "ఈ పాట గురించి విజయాలు, సవాళ్లు, మరియు ప్రేరణ గురించి ఒక పూర్తి పాట రాయండి. పాట ప్రారంభం:"
]

# Randomly select a prompt from the list
prompt = random.choice(prompts)

# Initialize the Transliterator for Telugu to Devanagari and vice versa
# Convert the prompt to Devanagari script
prompt_devanagari = UnicodeIndicTransliterator.transliterate(prompt, 'tel', 'hin')
prompt_devanagari = f"{prompt_devanagari} </s> <2te>"
print("Prompt in Devanagari:", prompt_devanagari)

# Tokenize the input prompt in Devanagari script
inputs = tokenizer(prompt_devanagari, return_tensors="pt")
print("Tokenized input IDs:", inputs.input_ids)
print("Token ID range:", inputs.input_ids.min().item(), "-", inputs.input_ids.max().item())
print("Model embedding layer configuration:", model.get_input_embeddings())

# Generate text using the model
bos_token_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_token_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
decoder_start_token_id = tokenizer._convert_token_to_id_with_added_voc("<2te>")  # Use Telugu token for generation
outputs = model.generate(
    inputs.input_ids,
    max_length=1500,  # Increase max_length to allow for longer sequences
    num_beams=5,
    early_stopping=False,
    bos_token_id=bos_token_id,
    eos_token_id=eos_token_id,
    decoder_start_token_id=decoder_start_token_id,
    forced_bos_token_id=decoder_start_token_id,
    no_repeat_ngram_size=3,
    num_return_sequences=5,
    repetition_penalty=2.0,  # Add repetition penalty to discourage repeated phrases
    length_penalty=1.0,  # Add length penalty to encourage longer sequences
    temperature=0.8,  # Adjust temperature to balance diversity and coherence
    top_k=30,  # Adjust top-k sampling to limit the number of highest probability tokens to keep for generation
    top_p=0.95,  # Adjust top-p (nucleus) sampling to keep the smallest set of tokens with cumulative probability >= top_p
    do_sample=True  # Enable sampling to allow temperature to take effect
)
print("Generated output IDs:", outputs)

# Decode the generated text
generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
print("Generated texts in Devanagari:", generated_texts)

# Convert the generated text back to Telugu script
generated_texts_telugu = [UnicodeIndicTransliterator.transliterate(text, 'hin', 'tel') for text in generated_texts]
print("Transliterated texts before filtering:", generated_texts_telugu)

# Filter out non-Telugu characters and placeholders from the generated text
filtered_texts_telugu = []
for text in generated_texts_telugu:
    filtered_text = ''.join([char for char in text if 0x0C00 <= ord(char) <= 0x0C7F or char in [' ', '.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}', '"', "'", '\n', '\t', '�']])
    # Remove placeholders and incomplete words
    filtered_text = filtered_text.replace('...', '')
    # Ensure the text is not empty and has a minimum length
    if len(filtered_text) > 10:
        filtered_texts_telugu.append(filtered_text)
print("Filtered texts in Telugu:", filtered_texts_telugu)

# Implement a content filter to check for inappropriate content
prohibited_words = ["గైంగ???ేప", "అశ్లీల", "అమానవీయ"]
final_texts_telugu = []
for text in filtered_texts_telugu:
    if not any(word in text for word in prohibited_words):
        final_texts_telugu.append(text)
    else:
        # Regenerate text if inappropriate content is found
        new_outputs = model.generate(
            inputs.input_ids,
            max_length=1500,
            num_beams=5,
            early_stopping=False,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_bos_token_id=decoder_start_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            repetition_penalty=2.0,
            length_penalty=1.0,
            temperature=1.2,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        new_generated_text = tokenizer.decode(new_outputs[0], skip_special_tokens=True)
        new_generated_text_telugu = UnicodeIndicTransliterator.transliterate(new_generated_text, 'hin', 'tel')
        new_filtered_text = ''.join([char for char in new_generated_text_telugu if 0x0C00 <= ord(char) <= 0x0C7F or char in [' ', '.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}', '"', "'", '\n', '\t']])
        final_texts_telugu.append(new_filtered_text)

# Print the generated songs
print("Generated Songs in Telugu:")
for idx, song in enumerate(final_texts_telugu):
    print(f"Song {idx + 1}:")
    print(song)
    print()

# Save the generated songs to a file
with open("generated_song_demo.txt", "w", encoding="utf-8") as f:
    for idx, song in enumerate(final_texts_telugu):
        f.write(f"Song {idx + 1}:\n")
        f.write(song + "\n\n")
