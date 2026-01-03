import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from gtts import gTTS
import os

# 1. Configuration & Model Loading
MODEL_NAME = "AhmedKElsayed/NonToxicMe"
LABEL_MAP = {
    0: "💀 Toxic", 1: "☠️ Very Toxic", 2: "🔞 Obscene",
    3: "⚠️ Threat", 4: "👊 Insult", 5: "🧭 Identity Hate"
}

# Load Toxicity Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load STT (Whisper) - Using tiny for speed
stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# 2. Logic Functions
def process_audio(audio_path):
    """Converts audio file to text."""
    if audio_path is None: return ""
    return stt_pipe(audio_path)["text"]

def generate_speech(text):
    """Converts prediction status to speech."""
    # Clean up emoji markers for better narration
    narration = text.replace("🚩", "Alert: ").replace("✅", "Status: ")
    tts = gTTS(text=narration, lang='en')
    filename = "result_voice.mp3"
    tts.save(filename)
    return filename

def analyze_and_voice(text_input, audio_input):
    # Use audio transcription if the text box is empty
    final_text = text_input if text_input.strip() else process_audio(audio_input)
    
    if not final_text.strip():
        return "No input provided.", None, gr.update(visible=False), ""

    # Tokenize and Predict
    inputs = tokenizer(final_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.sigmoid(logits)[0]
    is_toxic = any(prob > 0.5 for prob in probs)
    
    status = "🚩 Toxic" if is_toxic else "✅ Non-Toxic"
    audio_verdict = generate_speech(f"Result is {status[2:]}")

    if is_toxic:
        confidences = {LABEL_MAP[i]: float(prob.item()) for i, prob in enumerate(probs)}
        return status, audio_verdict, gr.update(value=confidences, visible=True), final_text
    
    return status, audio_verdict, gr.update(visible=False), final_text

# 3. Gradio Blocks Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ☢️ Non-Toxic Me")
    
    with gr.Row():
        with gr.Column():
            txt_in = gr.Textbox(label="Type Text", placeholder="How's your day going?")
            aud_in = gr.Audio(label="Or Speak", type="filepath")
            submit_btn = gr.Button("Run Analysis", variant="primary")
            
        with gr.Column():
            status_out = gr.Textbox(label="Analysis Status")
            voice_out = gr.Audio(label="Voice Feedback", autoplay=True)
            transcription_preview = gr.Markdown(label="Recognized Text")
            
            breakdown_out = gr.Label(
                label="Detailed Toxicity Levels",
                visible=False
            )

    submit_btn.click(
        fn=analyze_and_voice,
        inputs=[txt_in, aud_in],
        outputs=[status_out, voice_out, breakdown_out, transcription_preview]
    )

if __name__ == "__main__":
    demo.launch()