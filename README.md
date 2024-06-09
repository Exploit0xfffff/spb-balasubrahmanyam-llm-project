# spb-balasubrahmanyam-llm-project

## Project Overview
This project aims to emulate the vocal style of the legendary Indian playback singer SP Balasubrahmanyam using a language model (LLM). The study involves collecting datasets on SP Balasubrahmanyam's career and vocal performances, training an LLM to generate songs based on his unique vocal characteristics, and analyzing the results. The findings demonstrate the potential of LLMs in capturing and replicating the vocal essence of iconic singers.

## Installation
To set up the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/kasinadhsarma/spb-balasubrahmanyam-llm-project.git
   cd spb-balasubrahmanyam-llm-project
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To generate a song in Telugu using the pre-trained model, run the following script:
```bash
python3 generate_song_indic_bart.py
```

## LLM Model
### Purpose
The LLM (Language Model) used in this project is designed to generate song lyrics that emulate the vocal style of SP Balasubrahmanyam. The model is fine-tuned on a dataset of Telugu song lyrics to capture the nuances of the language and the singer's unique vocal characteristics.

### Model Details
- **Model Name:** ai4bharat/IndicBART
- **Tokenizer:** ./albert-indic64k
- **Checkpoint:** ./model_output/pytorch_model.bin
- **Language Token:** `<2te>` for Telugu

### Fine-Tuning
The model is fine-tuned using the `llm_fine_tuning.py` script with the `telugu_lyrics_dataset.txt` dataset. The fine-tuning process involves training the model on the dataset to improve its ability to generate coherent and contextually relevant song lyrics in Telugu.

### Parameters
- **Max Length:** 1000
- **Temperature:** 1.5
- **Early Stopping:** False
- **No Repeat Ngram Size:** 2
- **Num Return Sequences:** 3

### Dependencies
- `torch`
- `transformers`
- `sentencepiece`
- `indic-nlp-library`
- `accelerate`
- `safetensors`

### PYTHONPATH
Ensure the PYTHONPATH is set to include the user's local packages:
```bash
export PYTHONPATH=/home/ubuntu/.local/lib/python3.10/site-packages
```

## Current Progress
As of now, the project is approximately 80% complete. The following tasks have been completed:
- Analyzed the provided data and reviewed the datasets.
- Attempted to clean the dataset with `clean_dataset.py`.
- Updated the `README.md` file with the currently completed percentage.
- Edited the `clean_dataset.py` script to ensure it correctly removes line numbers.
- Verified the contents of the `cleaned_spb_texts.txt` file to ensure the data is correctly formatted.
- Drafted the research paper in IEEE format.
- Created and pushed the LLM fine-tuning script to the repository.
- Started and completed the fine-tuning process for the LLM using the `formatted_spb_texts.txt` dataset.
- Generated new song demos in Telugu using the `generate_song_indic_bart.py` script.

## Future Work
The following tasks are still pending:
- Upload all datasets to GitHub, pre-train the LLM, and deploy the LLM model.
- Find an alternative pre-trained model suitable for fine-tuning on SP Balasubrahmanyam's songs.
- Ensure generated songs accurately capture multiple languages.
- Request user assistance to resolve persistent shell and browser timeout issues.
- Continue refining the LLM to generate songs that accurately emulate SP Balasubrahmanyam's voice.
- Upload the refined song demos to GitHub after reaching 100% completion.

## References
- SP Balasubrahmanyam's Wikipedia page
- SP Balasubrahmanyam's official YouTube channel
- Song lyrics websites (Genius, Gaana, Musixmatch, tamil2lyrics.com, Smule)
- `transformers` library documentation: https://huggingface.co/transformers/
- `torch` library documentation: https://pytorch.org/
- `beautifulsoup4` library documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- `requests` library documentation: https://docs.python-requests.org/en/latest/
- `accelerate` library documentation: https://huggingface.co/docs/accelerate/index
