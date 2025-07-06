# AI-Meeting-Summarizer
This project presents an AI-powered solution for automating online meeting transcription, summarization, and action item extraction. It addresses the common challenge of missed or forgotten meeting content by generating concise, structured outputs from raw audio. Designed for professionals working in remote or hybrid environments, the tool improves meeting recall and reduces the need for manual follow-ups. Built with accessibility and clarity in mind, it allows users to quickly catch up on key decisions and responsibilities.
<br>
# Key Technologies Used
- **Python:** Core programming language used to integrate models and UI components.
- **Whisper:** Used for transcribing online meeting audio.
- **BART(HuggingFace Transformer):** Applied for abstractive summarization of the transcribed text.
- **Custom Regex-based NLP:** Used to extract action items assigned to users from the transcribed text using commonly used keywords in online meetings.
- **Streamlit:** Built an interactive frontend to display downloadable transcripts, summaries, and tasks.
<br>
# Project Workflow
- **1. File Upload:** The user uploads a recorded online meeting audio/video file (e.g., `.mp3`, `.wav`, `.mp4`).
- **2. Transcription with Whisper:** The uploaded audio is transcribed into raw text using OpenAI's Whisper model, with chunk-based processing for large files.
- **3. Summarization using BART:** The transcribed text is divided and processed in manageable segments by the BART transformer model to generate a concise, abstractive summary enabling support for longer transcripts without loss of context.
- **4. Action Item Extraction with Custom Regex-based NLP:** Handcrafted regex patterns, common during online meetings, are applied to the transcript to identify and extract assigned tasks and action items.
- **5. Display via Streamlit UI:** The final transcript, summary, and action items are displayed on a clean, interactive Streamlit frontend for user review along with downloading capability.
<br>
# Installation & Dependencies
