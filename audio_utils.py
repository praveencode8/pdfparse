import whisper
import tempfile
from config import GOOGLE_API_KEY
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load Whisper model once
whisper_model = whisper.load_model("medium")

# Gemini prompt setup
gemini_prompt_template = """
You are a smart sales assistant analyzing a customer call transcript.

Instructions:
- Summarize the conversation in a structured, sales/marketing-oriented format.
- Use **numbered bullet points** like this: 1), 2), 3)...
- Focus on points such as customer objections, interest areas, next steps (e.g., send brochure, follow-up), product mentions, etc.
- At the end, add a one-line overall sentiment summary in the format:
  **Sentiment:** Positive/Negative/Neutral — plus a reason.

Transcript:
\"\"\"{transcript}\"\"\"

Respond like this:

Summary:
1) ...
2) ...
3) ...
...
**Sentiment:** ...
"""


# Prepare LangChain components
gemini_prompt = PromptTemplate(template=gemini_prompt_template, input_variables=["transcript"])
gemini_model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
gemini_chain = LLMChain(llm=gemini_model, prompt=gemini_prompt)

# Process a single audio file
def process_single_audio(uploaded_file):
    # Transcribe audio with Whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        result = whisper_model.transcribe(tmp.name)
        transcript = result['text'].strip()

    # Summarize and get sentiment using Gemini
    response = gemini_chain.run({"transcript": transcript})

    # Separate summary and sentiment
    lines = response.strip().splitlines()
    summary_lines = [line for line in lines if not line.lower().startswith("**sentiment:**")]
    sentiment_line = next((line for line in lines if line.lower().startswith("**sentiment:**")), "Sentiment not found")

    return {
        "transcript": transcript,
        "summary": "\n".join(summary_lines),
        "sentiment": sentiment_line.replace("**Sentiment:**", "").strip()
    }

# Analyze multiple audio files
def analyze_audio_files(audio_files):
    results = [process_single_audio(file) for file in audio_files]
    similarity_matrix = None
    return results, similarity_matrix
