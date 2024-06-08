from transformers import pipeline
import soundfile as sf
from datasets import load_dataset
import torch

# Load the dataset containing speaker embeddings
dataset = load_dataset("Matthijs/cmu-arctic-xvectors")

# Assuming we need the first speaker's embeddings for the demo
speaker_embeddings = torch.tensor(dataset['validation'][0]['xvector']).unsqueeze(0)

# Initialize the text-to-speech pipeline with the SpeechT5 model
synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")

# Text input for the model
text_input = "Hello, this is a test to synthesize speech using the SpeechT5 model."

# Generate speech from text with speaker embeddings
speech = synthesizer(text_input, forward_params={"speaker_embeddings": speaker_embeddings})

# Save the generated speech to a WAV file
sf.write("test_speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
