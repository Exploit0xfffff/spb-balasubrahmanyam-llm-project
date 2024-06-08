from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
import torch

# Load the tokenizer and model from the local files
tokenizer = AutoTokenizer.from_pretrained("./albert-indic64k", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")

# Load the model weights from the checkpoint file
checkpoint = torch.load("./separate_script_indicbart_model.ckpt", map_location=torch.device('cpu'))
model.resize_token_embeddings(64015)  # Adjust the model's vocabulary size to match the checkpoint
model.load_state_dict(checkpoint, strict=False)

# Define the prompt for generating a song in Telugu
prompt = "తెలుగు పాట </s> <2te>"

# Convert the Telugu prompt to Devanagari script
prompt_devanagari = UnicodeIndicTransliterator.transliterate(prompt, "te", "hi")
print("Prompt in Devanagari script:", prompt_devanagari)

# Tokenize the input prompt
inputs = tokenizer(prompt_devanagari, return_tensors="pt")
print("Tokenized input IDs:", inputs.input_ids)

# Generate text using the model
bos_token_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_token_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
decoder_start_token_id = tokenizer._convert_token_to_id_with_added_voc("<2te>")  # Use Telugu token for Devanagari script
outputs = model.generate(inputs.input_ids, max_length=150, num_beams=10, early_stopping=True, bos_token_id=bos_token_id, eos_token_id=eos_token_id, decoder_start_token_id=decoder_start_token_id)
print("Generated output IDs:", outputs)

# Decode the generated text
generated_text_devanagari = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text in Devanagari script:", generated_text_devanagari)

# Convert the generated text back to Telugu script
generated_text = UnicodeIndicTransliterator.transliterate(generated_text_devanagari, "hi", "te")
print("Generated text in Telugu script:", generated_text)

# Print the generated song
print("Generated Song in Telugu:")
print(generated_text)
