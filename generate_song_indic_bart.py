from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART")
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")

# Define the prompt for generating a song in Telugu
prompt = "తెలుగు పాట"

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text using the model
outputs = model.generate(inputs.input_ids, max_length=100, num_beams=5, early_stopping=True)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated song
print("Generated Song in Telugu:")
print(generated_text)
