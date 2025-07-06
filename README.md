# AI-Meeting-Summarizer
This project presents an AI-powered solution for automating online meeting transcription, summarization, and action item extraction. It addresses the common challenge of missed or forgotten meeting content by generating concise, structured outputs from raw audio. Designed for professionals working in remote or hybrid environments, the tool improves meeting recall and reduces the need for manual follow-ups. Built with accessibility and clarity in mind, it allows users to quickly catch up on key decisions and responsibilities.
<br>
# Key Technologies Used
- **Python:** Core programming language used to integrate models and UI components.
- **Whisper:** Used for transcribing online meeting audio.
- **BART(HuggingFace Transformer):** Applied for abstractive summarization of the transcribed text.
- **Custom Regex-based NLP:** Used to extract action items assigned to users from the transcribed text using commonly used keywords in online meetings.
- **Streamlit:** Built an interactive frontend to display downloadable transcripts, summaries, and tasks.
