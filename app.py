import os
import re
import tempfile
import streamlit as st
import whisper
from pydub import AudioSegment
os.environ["USE_TF"] = "0"        # Disabled tensorflow as torch tensors perform same task with less overhead resulting in faster start up times (reasoned through experimentation)
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

@st.cache_resource
def load_summarization_model():
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1, torch_dtype=torch.float32)
        return summarizer
    except Exception as e:
        st.error(f"Failed to load summarization model: {e}")
        return None

def process_media(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    if tmp_path.endswith(('.mp4', '.mov', '.avi')):
        audio = AudioSegment.from_file(tmp_path).set_frame_rate(16000)
        audio_path = tmp_path + ".wav"
        audio.export(audio_path, format="wav")
        os.unlink(tmp_path)
    else:
        audio_path = tmp_path
    return audio_path

def chunk_audio(audio_path, chunk_size_ms=600000, overlap_ms=5000):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    start = 0
    while start < len(audio):
        end = start + chunk_size_ms
        chunk = audio[start:end]
        chunks.append(chunk)
        start = end - overlap_ms
    return chunks

def transcribe_long_audio(model, audio_path):
    chunks = chunk_audio(audio_path)
    full_transcript = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, chunk in enumerate(chunks):
        status_text.text(f"Processing chunk {i+1}/{len(chunks)}")
        progress_bar.progress((i + 1) / len(chunks))
        
        chunk_path = f"temp_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        result = model.transcribe(chunk_path)
        full_transcript.append(result["text"])
        os.unlink(chunk_path)
    
    progress_bar.empty()
    status_text.empty()
    return "\n".join(full_transcript)

def transcribe_audio(audio_path):
    model = load_whisper_model()
    audio = AudioSegment.from_file(audio_path)
    return transcribe_long_audio(model, audio_path) if len(audio) > 600000 else model.transcribe(audio_path)["text"]

def chunk_text_for_summarization(text, max_chunk_size=1000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def get_summary(text, summarizer=None):
    if not text.strip():
        return "No content to summarize."
    
    if summarizer is None:
        # Fallback
        return get_extractive_summary(text)
    
    try:
        # Chunk"s" the text to prevent limit on input size of model
        chunks = chunk_text_for_summarization(text, max_chunk_size=900)
        if not chunks:
            return "No content to summarize."
        summaries = []
        for i, chunk in enumerate(chunks):
            if len(chunk.split()) < 50:
                continue
            try:
                summary = summarizer(chunk, 
                                   max_length=130, 
                                   min_length=30, 
                                   do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                st.warning(f"Skipping chunk {i+1} due to processing error")
                continue
            
        if not summaries:
            return get_extractive_summary(text)
        combined_summary = "\n\n".join(f"• {summary}" for summary in summaries)
        return combined_summary
            
    except Exception as e:
        st.warning(f"AI summarization failed, using fallback method")
        return get_extractive_summary(text)

def get_extractive_summary(text):
    st.markdown("Generated Summary using important keyword matching (no model used).")
    sentences = sent_tokenize(text)
    if not sentences:
        return "No summary available."
    # Normally used keyword matching
    keywords = ['integration', 'issue', 'problem', 'solution', 'update', 'feature','bug', 'fix', 'improvement', 'release', 'testing', 'review','deadline', 'priority', 'critical', 'important', 'decided', 'agreed','API', 'dashboard', 'payment', 'merchant', 'validation', 'error']
    scored_sentences = []
    for sentence in sentences:
        score = sum(1 for keyword in keywords if keyword.lower() in sentence.lower())
        if score > 0 and len(sentence.split()) > 8:  # Minimum sentence length
            scored_sentences.append((score, sentence))
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [sent for score, sent in scored_sentences[:6]]
    if not top_sentences:
        return "Key discussion points could not be extracted automatically."
    
    return "\n\n".join(f"• {sent.strip()}" for sent in top_sentences)

def extract_actions(transcript):
    actions = []
    # Pattern 1: Direct assignments with names ex- "John, please do X", "Emily, prepare Y"
    pattern1 = re.findall(r'([A-Z][a-z]+),\s*(?:please\s+|kindly\s+)?([^.!?]+(?:by|before|after|until)[^.!?]*[.!?])',transcript, flags=re.IGNORECASE)
    for name, task in pattern1:
        clean_task = task.strip().rstrip('.!?')
        if len(clean_task.split()) > 3:
            actions.append(f"{name}: {clean_task}")
    # Pattern 2: "Name, action verb + task"
    pattern2 = re.findall(r'([A-Z][a-z]+),\s+((?:create|prepare|update|fix|review|send|contact|analyze|build|test|coordinate|finalize|draft|raise|investigate|add|design|schedule|follow up|reach out|talk to|notify|inform|check|audit|extract|include|optimize)[^.!?]*[.!?])',transcript, flags=re.IGNORECASE)
    for name, task in pattern2:
        clean_task = task.strip().rstrip('.!?')
        if len(clean_task.split()) > 2:
            actions.append(f"{name}: {clean_task}")
    # Pattern 3: Passive assignments "X needs to be done by Y"
    pattern3 = re.findall(r'([^.!?]+(?:needs to be|should be|must be)[^.!?]+by\s+([A-Z][a-z]+)[^.!?]*[.!?])',transcript, flags=re.IGNORECASE)
    for task_full, name in pattern3:
        clean_task = task_full.strip().rstrip('.!?')
        if len(clean_task.split()) > 4:
            actions.append(f"{name}: {clean_task}")
    # Pattern 4: "Name should/will/can + action"
    pattern4 = re.findall(r'([A-Z][a-z]+)\s+(?:should|will|can|must)\s+([^.!?]+[.!?])',transcript, flags=re.IGNORECASE)
    for name, task in pattern4:
        clean_task = task.strip().rstrip('.!?')
        if len(clean_task.split()) > 3 and 'meeting' not in clean_task.lower():
            actions.append(f"{name}: {clean_task}")
    
    # Cleaning and deduplicate
    unique_actions = []
    seen = set()
    for action in actions:
        # Basic cleaning
        action = re.sub(r'\s+', ' ', action.strip())
        action = action.replace(' ,', ',').replace(' .', '.')
        # Skip if too similar to existing actions
        action_key = action.lower()[:50]  # First 50 chars for similarity check
        if action_key not in seen and len(action.split()) > 3:
            seen.add(action_key)
            unique_actions.append(action)

    return unique_actions[:15]

def format_actions(actions):
    if not actions:
        return "No specific action items identified."
    formatted = []
    for i, action in enumerate(actions, 1):
        if ':' in action:
            name, task = action.split(':', 1)
            formatted.append(f"{i}. **{name.strip()}**: {task.strip()}")
        else:
            formatted.append(f"{i}. {action}")
    
    return "\n".join(formatted)

def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'file' not in st.session_state:
        st.session_state.file = None
    if 'done' not in st.session_state:
        st.session_state.done = False

def main():
    st.set_page_config(page_title="Meeting Summarizer", page_icon="🎙️", layout="wide")
    st.title("🎙️ AI Meeting Summarizer")
    st.markdown("Upload your meeting audio/video to get intelligent transcription, summary, and action items.")
    initialize_session_state()
    summarizer = load_summarization_model()
    if summarizer:
        st.success("✅ AI summarization model loaded successfully")
    else:
        st.warning("⚠️ AI model failed to load. Using fallback summarization.")
    
    uploaded_file = st.file_uploader("Upload meeting audio/video", type=["mp3", "wav", "mp4", "mov", "avi"],help="Supported formats: MP3, WAV, MP4, MOV, AVI")
    # Check if a new file is uploaded
    if uploaded_file:
        # Resets processing state if new file is uploaded
        if st.session_state.file != uploaded_file.name:
            st.session_state.data = None
            st.session_state.file = uploaded_file.name
            st.session_state.done = False
        
        st.audio(uploaded_file) # Preview
        
        # Shows process button only if not processed yet to avoid reproccessing during reruns
        if not st.session_state.done:
            if st.button("🚀 Process Meeting", type="primary"):
                try:
                    with st.spinner("🎵 Transcribing audio... This may take a few minutes depending on file size."):
                        audio_path = process_media(uploaded_file)
                        transcript = transcribe_audio(audio_path)
                        st.success("✅ Transcription completed!")
                    
                    with st.spinner("📝 Generating intelligent summary..."):
                        summary = get_summary(transcript, summarizer)
                        st.success("✅ Summary generated!")

                    with st.spinner("✅ Extracting action items..."):
                        actions = extract_actions(transcript)
                        actions = format_actions(actions)
                        st.success("✅ Action items extracted!")
                    
                    # storing data in session state to access across reruns
                    st.session_state.data = {
                        'transcript': transcript,
                        'summary': summary,
                        'actions': actions,
                        'filename': uploaded_file.name.split('.')[0]
                    }
                    st.session_state.done = True
                    os.unlink(audio_path) # Cleaning
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing meeting: {str(e)}")
                    st.info("Please try with a different file or check the file format.")
        
        # Output: Showing processed data
        if st.session_state.done and st.session_state.data:
            st.success("🎉 Meeting processed successfully! Check the tabs below.")
            data = st.session_state.data
            
            # Add a reset button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🔄 Process New File", type="secondary"):
                    st.session_state.data = None
                    st.session_state.done = False
                    st.session_state.file = None
                    st.rerun()
            
            tab1, tab2, tab3 = st.tabs(["📄 Transcript", "📋 Summary", "✅ Action Items"])
            with tab1:
                st.subheader("Full Transcript")
                st.text_area("", data['transcript'], height=300, label_visibility="collapsed")
                st.download_button("📥 Download Transcript",data['transcript'],file_name=f"transcript_{data['filename']}.txt",mime="text/plain",key="download_transcript")
            
            with tab2:
                st.subheader("Meeting Summary")
                st.markdown(data['summary'])
                st.download_button("📥 Download Summary",data['summary'],file_name=f"summary_{data['filename']}.txt",mime="text/plain",key="download_summary")
            
            with tab3:
                st.subheader("Action Items")
                st.markdown(data['actions'])
                st.download_button("📥 Download Action Items",data['actions'],file_name=f"action_items_{data['filename']}.txt",mime="text/plain",key="download_actions")

if __name__ == "__main__":
    main()