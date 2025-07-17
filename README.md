# AI-Meeting-Summarizer
This project presents an AI-powered solution for automating online meeting transcription, summarization, and action item extraction. It addresses the common challenge of missed or forgotten meeting content by generating concise, structured outputs from raw audio. Designed for professionals working in remote or hybrid environments, the tool improves meeting recall and reduces the need for manual follow-ups. Built with accessibility and clarity in mind, it allows users to quickly catch up on key decisions and responsibilities.
<br><br>
![Screenshot of starting page 1.](/StartingUI_1.png)
<br>
![Screenshot of starting page 1.](/StartingUI_2.png)
## Key Technologies Used
- **Python:** Core programming language used to integrate models and UI components.
- **Whisper:** Used for transcribing online meeting audio.
- **BART(HuggingFace Transformer):** Applied for abstractive summarization of the transcribed text.
- **Custom Regex-based NLP:** Used to extract action items assigned to users from the transcribed text using commonly used keywords in online meetings.
- **Streamlit:** Built an interactive frontend to display downloadable transcripts, summaries, and tasks.
## Project Workflow
1. **File Upload:** The user uploads a recorded online meeting audio/video file (e.g., `.mp3`, `.wav`, `.mp4`).
2. **Transcription with Whisper:** The uploaded audio is transcribed into raw text using OpenAI's Whisper model, with chunk-based processing for large files.
3. **Summarization using BART:** The transcribed text is divided and processed in manageable segments by the BART transformer model to generate a concise, abstractive summary, enabling support for longer transcripts without loss of context.
4. **Action Item Extraction with Custom Regex-based NLP:** Handcrafted regex patterns, common during online meetings, are applied to the transcript to identify and extract assigned tasks and action items.
5. **Display via Streamlit UI:** The final transcript, summary, and action items are displayed on a clean, interactive Streamlit frontend for user review, along with downloading capability.
> [!Note]
> I deployed the project locally using Streamlit as hosting was not feasible due to the large sizes of libraries and models used (>5GB storage) :(
## Installation & Dependencies
1. **Create a Virtual Environment:** I recommend doing so to avoid inconsistencies between different library versions. Use the command `conda create -n meet_env python=3.10.18` to create a virtual env and use `conda activate meet_env` to activate it.
2. **Clone the repository:** Use  `git clone _link_` to clone it to your local machine. Move to the repo's directory.
3. **Install all Dependencies:** Use `conda install -r requirements.txt` to install libraries. Also, manually download the 'punkt' module of nltk using `nltk.download('punkt')`
4. **Run the Streamlit App:** Use `Streamlit run app.py` to run the app.
> [!Note]
> During 1st run, it will take some time to download the models used.
## Use Cases
- Employees who miss meetings often struggle to catch up or ask colleagues, leading to miscommunication and delays.
- Taking notes manually during meetings is time-consuming and prone to missing important details.
- This tool allows users to quickly revisit key decisions and action items without relying on others.
<br><br> `Refer to In_Between_Processing folder for processing UI images.`<br>


https://github.com/user-attachments/assets/bed84744-be09-4e0d-a8c1-9380f84edd37

`Refer to the Outputs folder for downloaded files`

## Future Improvements
- Enable real-time transcription support for live meetings by integrating advanced streaming-capable and speaker identification models or deploying on a dedicated server.
- Expand language support to handle multilingual meetings and non-English transcripts.
- Implement Named Entity Recognition (NER) to highlight key names, dates, tools, or organizations mentioned.
