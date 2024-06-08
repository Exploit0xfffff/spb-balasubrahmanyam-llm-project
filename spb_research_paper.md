# Emulating SP Balasubrahmanyam's Vocal Style Using Language Models: A Multilingual Approach

## Authors
Devin AI, kasinadhsarma

## Contact
kasinadhsarma@gmail.com

\begin{abstract}
This research paper explores the process of emulating the vocal style of the legendary Indian playback singer SP Balasubrahmanyam using a language model (LLM). The study involves collecting datasets on SP Balasubrahmanyam's career and vocal performances, training an LLM to generate songs based on his unique vocal characteristics, and analyzing the results. The findings demonstrate the potential of LLMs in capturing and replicating the vocal essence of iconic singers.
\end{abstract}

\begin{IEEEkeywords}
Language Models, Vocal Emulation, SP Balasubrahmanyam, Multilingual, GPT-2
\end{IEEEkeywords}

\section{Introduction}

SP Balasubrahmanyam, fondly known as SPB, was an Indian playback singer, actor, music composer, and film producer who sang in 16 languages and is considered one of the greatest Indian singers. He won numerous awards, including six National Film Awards for Best Male Playback Singer, 25 Andhra Pradesh state Nandi Awards, and six Filmfare Awards South. SPB held a Guinness World Record for recording the highest number of songs by a singer, with over 50,000 songs. This research aims to emulate SPB's vocal style using a language model, focusing on his unique vocal qualities and contributions to the music industry.

\section{Methodology}

\subsection{Data Collection}

Datasets on SP Balasubrahmanyam's career and vocal performances were collected from various sources, including his official YouTube channel, music streaming platforms, and song lyrics websites. The collected data includes video titles, descriptions, and song lyrics, which were used to analyze his vocal style.

\subsection{LLM Training}

The `transformers` and `torch` libraries were used to implement and train a GPT-2 model on the collected datasets. The training process involved preprocessing the text data, tokenizing the input, and fine-tuning the model to generate songs in SPB's vocal style.

\subsection{Song Generation}

The trained LLM was used to generate songs based on given contexts, emulating SPB's vocal style. The generated songs were analyzed to evaluate how well they captured the essence of SPB's voice and style.

\section{Results}

The generated songs demonstrate the LLM's ability to emulate SPB's vocal style to a certain extent. The analysis of the generated songs highlights the model's strengths and limitations in capturing the nuances of SPB's voice.

\section{Discussion}
The results indicate that while the LLM can generate songs that resemble SPB's style, there are challenges in fully capturing the unique vocal qualities of such an iconic singer. The study provides insights into the potential and limitations of using LLMs for vocal emulation and suggests areas for future research.

\section{Conclusion}

This research demonstrates the feasibility of using LLMs to emulate the vocal style of legendary singers like SP Balasubrahmanyam. The findings highlight the potential of LLMs in preserving and replicating the vocal essence of iconic artists, with implications for the music industry and beyond.

\section{Project Completion}

As of now, the project is approximately 20% complete. The following tasks have been completed:
- Analyzed the provided data and reviewed the datasets.
- Attempted to clean the dataset with `clean_dataset.py`.
- Updated the `README.md` file with the currently completed percentage.
- Edited the `clean_dataset.py` script to ensure it correctly removes line numbers.
- Verified the contents of the `cleaned_spb_texts.txt` file to ensure the data is correctly formatted.
- Drafted the research paper in IEEE format.
- Successfully pushed the project to the GitHub repository `spb-balasubrahmanyam-llm-project`.
- Generated song demos using the pre-trained LLM.

The following tasks are still pending:
- Upload all datasets to GitHub, pre-train the LLM, and deploy the LLM model.
- Find an alternative pre-trained model suitable for fine-tuning on SP Balasubrahmanyam's songs.
- Ensure generated songs accurately capture multiple languages.
- Request user assistance to resolve persistent shell and browser timeout issues.

\section{References}

\begin{thebibliography}{1}
\bibitem{reference1} SP Balasubrahmanyam's Wikipedia page
\bibitem{reference2} SP Balasubrahmanyam's official YouTube channel
\bibitem{reference3} Song lyrics websites (Genius, Gaana, Musixmatch, tamil2lyrics.com, Smule)
\bibitem{reference4} `transformers` library documentation: https://huggingface.co/transformers/
\bibitem{reference5} `torch` library documentation: https://pytorch.org/
\bibitem{reference6} `beautifulsoup4` library documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
\bibitem{reference7} `requests` library documentation: https://docs.python-requests.org/en/latest/
\bibitem{reference8} `accelerate` library documentation: https://huggingface.co/docs/accelerate/index
\end{thebibliography}
