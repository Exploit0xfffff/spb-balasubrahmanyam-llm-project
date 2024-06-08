from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indictrans import Transliterator
import torch

# Load the tokenizer and model from the local files
tokenizer = AutoTokenizer.from_pretrained("./albert-indic64k", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")

# Load the model weights from the checkpoint file
checkpoint = torch.load("./separate_script_indicbart_model.ckpt", map_location=torch.device('cpu'))
model.resize_token_embeddings(64015)  # Adjust the model's vocabulary size to match the checkpoint
model.load_state_dict(checkpoint, strict=False)

# Define the prompt for generating a song in Telugu
prompt = "ఈ పాట గురించి ప్రేమ, బాధ, ఆనందం, మరియు జీవితం గురించి ఒక పూర్తి పాట రాయండి. పాట ప్రారంభం:"

# Initialize the Transliterator for Telugu to Devanagari and vice versa
transliterator_te_to_hi = Transliterator(source='tel', target='hin', build_lookup=True)
transliterator_hi_to_te = Transliterator(source='hin', target='tel', build_lookup=True)

# Convert the prompt to Devanagari script
prompt_devanagari = transliterator_te_to_hi.transform(prompt)
prompt_devanagari = f"{prompt_devanagari} </s> <2te>"

# Tokenize the input prompt in Devanagari script
inputs = tokenizer(prompt_devanagari, return_tensors="pt")
print("Tokenized input IDs:", inputs.input_ids)

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
    num_return_sequences=3,
    repetition_penalty=2.0,  # Add repetition penalty to discourage repeated phrases
    length_penalty=1.0,  # Add length penalty to encourage longer sequences
    temperature=1.2,  # Adjust temperature to balance diversity and coherence
    top_k=50,  # Adjust top-k sampling to limit the number of highest probability tokens to keep for generation
    top_p=0.9,  # Adjust top-p (nucleus) sampling to keep the smallest set of tokens with cumulative probability >= top_p
    do_sample=True  # Enable sampling to allow temperature to take effect
)
print("Generated output IDs:", outputs)

# Decode the generated text
generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
print("Generated texts in Devanagari:", generated_texts)

# Convert the generated text back to Telugu script
generated_texts_telugu = [transliterator_hi_to_te.transform(text) for text in generated_texts]
print("Transliterated texts before filtering:", generated_texts_telugu)

# Filter out non-Telugu characters from the generated text
filtered_texts_telugu = []
for text in generated_texts_telugu:
    filtered_text = ''.join([char for char in text if 0x0C00 <= ord(char) <= 0x0C7F or char in [' ', '.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}', '"', "'", '\n', '\t']])
    filtered_texts_telugu.append(filtered_text)
print("Filtered texts in Telugu:", filtered_texts_telugu)

# Implement a content filter to check for inappropriate content
prohibited_words = ["గైంగరేప", "అశ్లీల", "అమానవీయ"]
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
        new_generated_text_telugu = transliterator_hi_to_te.transform(new_generated_text)
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
