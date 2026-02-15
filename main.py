import subprocess
import os
import gradio as gr
import requests
import json
import time
import tempfile
from datetime import datetime

OLLAMA_SERVER_URL = "http://localhost:11434"  # Replace this with your actual Ollama server URL if different
WHISPER_MODEL_DIR = "./whisper.cpp/models"  # Directory where whisper models are stored


def get_available_models() -> list[str]:
    """
    Retrieves a list of all available models from the Ollama server and extracts the model names.

    Returns:
        A list of model names available on the Ollama server.
    """
    response = requests.get(f"{OLLAMA_SERVER_URL}/api/tags")
    if response.status_code == 200:
        models = response.json()["models"]
        llm_model_names = [model["model"] for model in models]  # Extract model names
        return llm_model_names
    else:
        raise Exception(
            f"Failed to retrieve models from Ollama server: {response.text}"
        )


def get_available_whisper_models() -> list[str]:
    """
    Retrieves a list of available Whisper models based on downloaded .bin files in the whisper.cpp/models directory.
    Filters out test models and only includes official Whisper models (e.g., base, small, medium, large).

    Returns:
        A list of available Whisper model names (e.g., 'base', 'small', 'medium', 'large-V3').
    """
    # List of acceptable official Whisper models
    valid_models = ["base", "small", "medium", "large", "large-V3"]

    # Get the list of model files in the models directory
    model_files = [f for f in os.listdir(WHISPER_MODEL_DIR) if f.endswith(".bin")]

    # Filter out test models and models that aren't in the valid list
    whisper_models = [
        os.path.splitext(f)[0].replace("ggml-", "")
        for f in model_files
        if any(valid_model in f for valid_model in valid_models) and "test" not in f
    ]

    # Remove any potential duplicates
    whisper_models = list(set(whisper_models))

    return whisper_models


def summarize_with_model(llm_model_name: str, context: str, text: str) -> str:
    """
    Uses a specified model on the Ollama server to generate a summary.
    Handles streaming responses by processing each line of the response.

    Args:
        llm_model_name (str): The name of the model to use for summarization.
        context (str): Optional context for the summary, provided by the user.
        text (str): The transcript text to summarize.

    Returns:
        str: The generated summary text from the model.
    """
    prompt = f"""You are given a transcript from a meeting, along with some optional context.
    
    Context: {context if context else 'No additional context provided.'}
    
    The transcript is as follows:
    
    {text}
    
    Please summarize the transcript."""

    headers = {"Content-Type": "application/json"}
    data = {"model": llm_model_name, "prompt": prompt}

    response = requests.post(
        f"{OLLAMA_SERVER_URL}/api/generate", json=data, headers=headers, stream=True
    )

    if response.status_code == 200:
        full_response = ""
        try:
            # Process the streaming response line by line
            for line in response.iter_lines():
                if line:
                    # Decode each line and parse it as a JSON object
                    decoded_line = line.decode("utf-8")
                    json_line = json.loads(decoded_line)
                    # Extract the "response" part from each JSON object
                    full_response += json_line.get("response", "")
                    # If "done" is True, break the loop
                    if json_line.get("done", False):
                        break
            return full_response
        except json.JSONDecodeError:
            print("Error: Response contains invalid JSON data.")
            return f"Failed to parse the response from the server. Raw response: {response.text}"
    else:
        raise Exception(
            f"Failed to summarize with model {llm_model_name}: {response.text}"
        )


def save_summary_txt(summary: str) -> str:
    """
    Save the summary text to a temporary TXT file and return its path.
    """
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="summary_")
    with os.fdopen(fd, "w") as f:
        f.write(summary)
    return path


def save_summary_pdf(summary: str) -> str:
    """
    Try to save the summary as a simple PDF. If fpdf is not available, raise.
    """
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in summary.split("\n"):
            pdf.multi_cell(0, 6, line)

        fd, path = tempfile.mkstemp(suffix=".pdf", prefix="summary_")
        os.close(fd)
        pdf.output(path)
        return path
    except Exception as e:
        raise


def preprocess_audio_file(audio_file_path: str) -> str:
    """
    Converts the input audio file to a WAV format with 16kHz sample rate and mono channel.

    Args:
        audio_file_path (str): Path to the input audio file.

    Returns:
        str: The path to the preprocessed WAV file.
    """
    output_wav_file = f"{os.path.splitext(audio_file_path)[0]}_converted.wav"

    # Ensure ffmpeg converts to 16kHz sample rate and mono channel
    cmd = f'ffmpeg -y -i "{audio_file_path}" -ar 16000 -ac 1 "{output_wav_file}"'
    subprocess.run(cmd, shell=True, check=True)

    return output_wav_file


def translate_and_summarize(
    audio_file_path: str, context: str, whisper_model_name: str, llm_model_name: str
) -> tuple[str, str]:
    """
    Translates the audio file into text using the whisper.cpp model and generates a summary using Ollama.
    Also provides the transcript file for download.

    Args:
        audio_file_path (str): Path to the input audio file.
        context (str): Optional context to include in the summary.
        whisper_model_name (str): Whisper model to use for audio-to-text conversion.
        llm_model_name (str): Model to use for summarizing the transcript.

    Returns:
        tuple[str, str]: A tuple containing the summary and the path to the transcript file for download.
    """
    output_file = "output.txt"

    print("Processing audio file:", audio_file_path)

    # Convert the input file to WAV format if necessary
    audio_file_wav = preprocess_audio_file(audio_file_path)

    print("Audio preprocessed:", audio_file_wav)

    # Call the whisper.cpp binary
    whisper_command = f'./whisper.cpp/main -m ./whisper.cpp/models/ggml-{whisper_model_name}.bin -f "{audio_file_wav}" > {output_file}'
    subprocess.run(whisper_command, shell=True, check=True)

    print("Whisper.cpp executed successfully")

    # Read the output from the transcript
    with open(output_file, "r") as f:
        transcript = f.read()

    # Save the transcript to a downloadable file
    transcript_file = "transcript.txt"
    with open(transcript_file, "w") as transcript_f:
        transcript_f.write(transcript)

    # Generate summary from the transcript using Ollama's model
    summary = summarize_with_model(llm_model_name, context, transcript)

    # Clean up temporary files
    os.remove(audio_file_wav)
    os.remove(output_file)

    # Return the downloadable link for the transcript and the summary text
    return summary, transcript_file


# New streaming processing function for richer UI feedback
def process_audio_stream(
    audio_file_path: str,
    context: str,
    whisper_model_name: str,
    llm_model_name: str,
    summary_mode: str = "Bullet Summary",
    detect_language: bool = True,
) -> tuple[str, int, str, str, str, str]:
    """
    Generator-style function to stream status updates to the UI.
    Yields tuples matching the UI outputs: (status_text, progress, summary_html, transcript_path)
    """
    try:
        # 1) Status: transcribing (show spinner)
        yield "Transcribing audio...", gr.update(value=10), "", None, "", "<div class='spinner'></div>"

        # Preprocess (convert) audio
        audio_file_wav = preprocess_audio_file(audio_file_path)
        yield "Running Whisper for speech-to-text...", gr.update(value=30), "", None, "", "<div class='spinner'></div>"

        # Run whisper.cpp binary
        output_file = "output.txt"
        whisper_command = f'./whisper.cpp/main -m ./whisper.cpp/models/ggml-{whisper_model_name}.bin -f "{audio_file_wav}" > {output_file}'
        subprocess.run(whisper_command, shell=True, check=True)

        # Read transcript
        with open(output_file, "r") as f:
            transcript = f.read()

        # Save transcript
        transcript_file = "transcript.txt"
        with open(transcript_file, "w") as transcript_f:
            transcript_f.write(transcript)

        # Progress update
        yield "Generating summary...", gr.update(value=60), "", transcript_file, transcript, "<div class='spinner'></div>"

        # Summarize using Ollama
        summary_raw = summarize_with_model(llm_model_name, context, transcript)

        # Post-process summary into sections based on simple heuristics
        summary_html = format_summary_html(summary_raw, summary_mode)

        # Finalizing
        yield "Finalizing...", gr.update(value=95), summary_html, transcript_file, transcript, "<div class='spinner'></div>"

        # Cleanup
        try:
            os.remove(audio_file_wav)
            os.remove(output_file)
        except Exception:
            pass

        yield "Done", gr.update(value=100), summary_html, transcript_file, transcript, ""

    except Exception as e:
        yield f"Error: {str(e)}", gr.update(value=0), "", None, "", ""


def format_summary_html(summary_text: str, mode: str = "Bullet Summary") -> str:
    """
    Create a presentable HTML block with sections: Key Points, Action Items, Decisions, Deadlines.
    Uses simple heuristics to split the model output. Keep this deterministic and safe.
    """
    # Basic heuristics: look for lines starting with Action:, Decide:, Deadline:
    lines = [l.strip() for l in summary_text.splitlines() if l.strip()]
    key_points = []
    actions = []
    decisions = []
    deadlines = []

    for ln in lines:
        low = ln.lower()
        if low.startswith("action") or "action item" in low:
            actions.append(ln)
        elif low.startswith("decision") or "decis" in low:
            decisions.append(ln)
        elif "deadline" in low or "due" in low:
            deadlines.append(ln)
        else:
            key_points.append(ln)

    # If mode asks for action items only
    if mode == "Action Items Only":
        body = "<h3>Action Items</h3>" + list_to_html(actions or key_points)
    elif mode == "Executive Summary":
        # Executive: show first 4 key points
        body = "<h3>Executive Summary</h3>" + list_to_html(key_points[:4])
    elif mode == "Detailed Summary":
        body = "<h3>Summary</h3>" + list_to_html(lines)
    else:
        # Bullet summary / default
        body = ""
        if key_points:
            body += "<h3>Key Points</h3>" + list_to_html(key_points)
        if actions:
            body += "<h3>Action Items</h3>" + list_to_html(actions)
        if decisions:
            body += "<h3>Decisions Made</h3>" + list_to_html(decisions)
        if deadlines:
            body += "<h3>Deadlines</h3>" + list_to_html(deadlines)

    # Wrap in a card-like container
    html = f"<div class=\"summary-card\">{body}</div>"
    return html


def list_to_html(items: list[str]) -> str:
    if not items:
        return "<p class='muted'>No items detected.</p>"
    html = "<ul>"
    for it in items:
        html += f"<li>{gradio_escape(it)}</li>"
    html += "</ul>"
    return html


def gradio_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('\n', '<br/>')
    )


def get_file_info(audio_filepath: str):
    """
    Given a local audio filepath, return (filename, duration_str).
    Uses ffprobe (part of ffmpeg) to query duration. If unavailable, returns empty duration.
    """
    if not audio_filepath:
        return None, None
    try:
        fname = os.path.basename(audio_filepath)
        # try ffprobe to get duration
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_filepath,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        dur = result.stdout.strip()
        if dur:
            secs = float(dur)
            m = int(secs // 60)
            s = int(secs % 60)
            dur_str = f"{m}:{s:02d}"
        else:
            dur_str = ""
        return fname, dur_str
    except Exception:
        return os.path.basename(audio_filepath), ""


# Gradio interface
def gradio_app(
    audio, context: str, whisper_model_name: str, llm_model_name: str
) -> tuple[str, str]:
    """
    Gradio application to handle file upload, model selection, and summary generation.

    Args:
        audio: The uploaded audio file.
        context (str): Optional context provided by the user.
        whisper_model_name (str): The selected Whisper model name.
        llm_model_name (str): The selected language model for summarization.

    Returns:
        tuple[str, str]: A tuple containing the summary text and a downloadable transcript file.
    """
    return translate_and_summarize(audio, context, whisper_model_name, llm_model_name)


# Main function to launch the Gradio interface
if __name__ == "__main__":
    # Retrieve available models for Gradio dropdown input
    try:
        ollama_models = get_available_models()  # Retrieve models from Ollama server
    except Exception:
        ollama_models = ["gpt-4o", "gpt-4o-mini"]

    whisper_models = get_available_whisper_models() or ["base", "small", "medium"]

    CSS = """
    <style>
    :root{--accent:#6C63FF;--bg1:linear-gradient(135deg,#0f172a 0%, #071029 100%);--card:#0b1220}
    body {font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;}
    .app-header{display:flex;align-items:center;gap:12px;padding:18px 12px;color:white}
    .logo{width:44px;height:44px;background:linear-gradient(135deg,var(--accent),#8b5cf6);border-radius:10px;box-shadow:0 6px 18px rgba(12,13,40,.6)}
    .tagline{font-size:14px;opacity:.9}
    .container{max-width:1200px;margin:18px auto}
    .card{background:rgba(255,255,255,0.03);border-radius:14px;padding:18px;box-shadow:0 8px 30px rgba(2,6,23,.6);border:1px solid rgba(255,255,255,0.04);color:#e6eef8}
    .summary-card{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));padding:16px;border-radius:10px}
    .muted{color:#9aa4bf}
    .controls .label{font-weight:600;margin-bottom:6px}
    .footer{margin-top:18px;padding:12px;color:#9aa4bf;text-align:center;font-size:13px}
    .upload-drop{transition:box-shadow .18s,transform .12s}
    .upload-drop.gradio-file-drop:hover{transform:translateY(-2px)}
    .spinner{width:36px;height:36px;border:4px solid rgba(255,255,255,0.08);border-top-color:var(--accent);border-radius:50%;animation:spin 1s linear infinite;margin:6px}
    @keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}
    .btn-accent{background:var(--accent);color:white}
    .btn-accent:hover{opacity:.95;transform:translateY(-1px)}
    @media (max-width:900px){.two-col{flex-direction:column}}
    </style>
    """

    with gr.Blocks(css=CSS, title="Meeting Summarizer - Modern UI") as demo:
        gr.HTML("<div class='container'>")

        # Header
        with gr.Row(elem_id="header", variant="huggingface"):
            with gr.Column(scale=1):
                gr.HTML("<div class='app-header'><div class='logo'></div><div><div style='font-weight:700'>Meeting Summarizer</div><div class='tagline'>Turn conversations into insights in seconds.</div></div></div>")

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(elem_classes="card"):
                    gr.Markdown("**Upload & Settings**")
                    audio_in = gr.Audio(label="Upload an audio file", type="filepath")
                    file_name = gr.Textbox(label="File name", interactive=False)
                    file_duration = gr.Textbox(label="Duration", interactive=False)
                    supported = gr.HTML("<div class='muted'>Supported: MP3, WAV, M4A — Max 50MB</div>")
                    context = gr.Textbox(label="Context (optional)", placeholder="Add meeting context to improve summary", lines=3)
                    with gr.Row():
                        whisper_dropdown = gr.Dropdown(choices=whisper_models, value=whisper_models[0], label="Whisper model")
                        ollama_dropdown = gr.Dropdown(choices=ollama_models, value=ollama_models[0] if ollama_models else None, label="LLM model")
                    summary_mode = gr.Radio(choices=["Bullet Summary", "Executive Summary", "Action Items Only", "Detailed Summary"], value="Bullet Summary", label="Summary Mode")
                    detect_language = gr.Checkbox(label="Auto-detect language", value=True)
                    with gr.Row():
                        generate_btn = gr.Button("Generate Summary", elem_classes="btn-accent")
                        reset_btn = gr.Button("Reset")
                        report_btn = gr.Button("Report Issue")

            with gr.Column(scale=6):
                with gr.Column(elem_classes="card"):
                    gr.Markdown("**Results**")
                    status = gr.Textbox(label="Status", interactive=False, value="Idle")
                    spinner_html = gr.HTML("<div class='spinner' style='display:none'></div>", visible=False)
                    progress = gr.Slider(minimum=0, maximum=100, value=0, interactive=False, label="Progress")

                    with gr.Tabs():
                        with gr.TabItem("Summary"):
                            summary_html = gr.HTML("<div class='muted'>No summary yet.</div>")
                            with gr.Row():
                                copy_btn = gr.Button("Copy to clipboard", elem_id="copy-btn")
                                download_txt = gr.Button("Download TXT")
                                download_pdf = gr.Button("Download PDF")
                                download_file_txt = gr.File(visible=False)
                                download_file_pdf = gr.File(visible=False)
                        with gr.TabItem("Transcript"):
                            transcript_text = gr.Textbox(label="Transcript", lines=12)
                            transcript_file = gr.File(label="Download Transcript")

        gr.HTML("</div>")

        # JS: Copy to clipboard using a small inline script bound to the button id
        gr.HTML("<script>document.addEventListener('click', e => { if(e.target && e.target.id==='copy-btn'){ const el=document.querySelector('.summary-card'); if(el) navigator.clipboard.writeText(el.innerText); } });</script>")

        # Generate logic: connect streaming function (includes spinner output)
        generate_btn.click(
            fn=process_audio_stream,
            inputs=[audio_in, context, whisper_dropdown, ollama_dropdown, summary_mode, detect_language],
            outputs=[status, progress, summary_html, transcript_file, transcript_text, spinner_html],
        )

        # Update file name and duration when an audio file is uploaded
        audio_in.change(fn=get_file_info, inputs=[audio_in], outputs=[file_name, file_duration])

        # Download handlers
        def export_txt(summary_html_str: str):
            # strip simple HTML tags for TXT export
            import re

            txt = re.sub('<[^<]+?>', '', summary_html_str or '')
            path = save_summary_txt(txt)
            return path

        def export_pdf(summary_html_str: str):
            import re

            txt = re.sub('<[^<]+?>', '', summary_html_str or '')
            try:
                path = save_summary_pdf(txt)
                return path
            except Exception:
                # If PDF generation fails, return None so UI can show nothing
                return None

        download_txt.click(fn=export_txt, inputs=[summary_html], outputs=[download_file_txt])
        download_pdf.click(fn=export_pdf, inputs=[summary_html], outputs=[download_file_pdf])

        # Reset button
        def do_reset():
            return None, "", "", "Idle", 0, "", None, "", ""

        reset_btn.click(fn=do_reset, inputs=None, outputs=[audio_in, file_name, file_duration, status, progress, summary_html, transcript_file, transcript_text, spinner_html])

    demo.launch(debug=True, share=True)
