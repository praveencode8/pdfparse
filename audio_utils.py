import whisper
import tempfile
from transformers import pipeline

# Load models
whisper_model = whisper.load_model("medium")  # You can choose another whisper model
summarizer = pipeline("summarization", model="facebook/bart-base")

def process_single_audio(uploaded_file):
    # Save uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()

        # Transcribe using Whisper model
        result = whisper_model.transcribe(tmp.name)
        transcript = result['text']

    # Summarize if transcript length is large
    summary = None
    if len(transcript.split()) > 100:
        summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    return {
        'transcript': transcript,
        'summary': summary
    }

def analyze_audio_files(audio_files):
    # Process all audio files and get the transcription and summary
    results = [process_single_audio(file) for file in audio_files]
    
    # Collect the summaries
    summaries = []
    for result in results:
        summaries.append(result['summary'] if result['summary'] else result['transcript'])
    
    return results, summaries
