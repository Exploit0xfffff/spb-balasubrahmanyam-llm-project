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

## Current Progress
As of now, the project is approximately 50% complete. The following tasks have been completed:
- Analyzed the provided data and reviewed the datasets.
- Attempted to clean the dataset with `clean_dataset.py`.
- Updated the `README.md` file with the currently completed percentage.
- Edited the `clean_dataset.py` script to ensure it correctly removes line numbers.
- Verified the contents of the `cleaned_spb_texts.txt` file to ensure the data is correctly formatted.
- Drafted the research paper in IEEE format.
- Created and pushed the LLM fine-tuning script to the repository.
- Started and completed the fine-tuning process for the LLM using the `formatted_spb_texts.txt` dataset.

## Future Work
The following tasks are still pending:
- Upload all datasets to GitHub, pre-train the LLM, and deploy the LLM model.
- Find an alternative pre-trained model suitable for fine-tuning on SP Balasubrahmanyam's songs.
- Ensure generated songs accurately capture multiple languages.
- Request user assistance to resolve persistent shell and browser timeout issues.
- Generate song demos using the fine-tuned model and upload them to GitHub after reaching 50% completion.
- Continue refining the LLM to generate songs that accurately emulate SP Balasubrahmanyam's voice.

## References
- SP Balasubrahmanyam's Wikipedia page
- SP Balasubrahmanyam's official YouTube channel
- Song lyrics websites (Genius, Gaana, Musixmatch, tamil2lyrics.com, Smule)
- `transformers` library documentation: https://huggingface.co/transformers/
- `torch` library documentation: https://pytorch.org/
- `beautifulsoup4` library documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- `requests` library documentation: https://docs.python-requests.org/en/latest/
- `accelerate` library documentation: https://huggingface.co/docs/accelerate/index
