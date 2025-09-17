import sys
import threading
import queue
import time
import os
from pathlib import Path
import subprocess
import json
import uuid
from dotenv import load_dotenv
import traceback
from flask import Flask, Response, render_template, request, stream_with_context, jsonify
from werkzeug.utils import secure_filename
from src.goldilocks_and_the_3_agents import run_this_thing

try:
    from flask import send_file, has_request_context
except Exception:
    # CLI-only environments won't have Flask loaded; that's fine.
    def has_request_context():  # type: ignore
        return False
    def send_file(*args, **kwargs):  # type: ignore
        raise RuntimeError("send_file is not available outside Flask.")

# Load environment variables from .env file
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not found in environment variables!")

app = Flask(__name__)

latest_form_data = {}
sessions = {}

def process_input(session_data):
    # Create a queue to collect output messages.
    q = queue.Queue()

    # Define a stream that writes to the queue.
    class StreamToQueue:
        def __init__(self, queue):
            self.queue = queue
            self.buffer = ""

        def put(self, msg):
            # Append new text to the buffer.
            self.buffer += msg + "\n"
            # If there is a newline in the buffer, split it into complete lines.
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                # Put the complete line into the queue.
                self.queue.put(line)

        def flush(self):
            # Flush any remaining text.
            if self.buffer:
                self.queue.put(self.buffer)
                self.buffer = ""

    # Run the long-running function in a separate thread
    def run_thread():
        # Create a local stdout redirector
        output_queue = StreamToQueue(q)

        try:
            run_this_thing(session_data, output_queue)
        except Exception:
            q.put("[ERROR]::: " + traceback.format_exc())
        finally:
            q.put("[END]::: Streaming complete.")

    thread = threading.Thread(target=run_thread, daemon=True)
    thread.start()

    last_sent = time.time()
    heartbeat_every = 15  # seconds

    while thread.is_alive() or not q.empty():
        try:
            msg = q.get(timeout=1.0)
            yield msg
            last_sent = time.time()
        except queue.Empty:
            # send a comment ping if idle for too long
            if time.time() - last_sent >= heartbeat_every:
                yield ""  # SSE comment (no 'data:'), keeps connection alive
                last_sent = time.time()

    thread.join()

@app.route("/stream/<path:session_id>")
def stream(session_id):
    # pull once; if missing, return SSE error event
    session_data = sessions.pop(session_id, None)
    if session_data is None:
        return Response("event: error\ndata: Invalid session\n\n",
                        mimetype="text/event-stream", status=404)

    def generate():
        last_ping = time.time()
        # stream model output
        for update in process_input(session_data):
            yield f"data: {update}\n\n"
            last_ping = time.time()

        # just in case, send a final END (your worker already does this)
        # yield "data: [END]::: Streaming complete.\n\n"

    # Important headers for SSE + proxies
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # for nginx
    }

    # Wrap generator to inject heartbeats
    def heartbeat_wrapper(gen, interval=15):
        last = time.time()
        for chunk in gen:
            yield chunk
            last = time.time()
        # generator finished; stop
        return
        # (If you want periodic pings while the generator is still running,
        # move this heartbeat logic inside process_input or use a side thread.)

    return Response(stream_with_context(heartbeat_wrapper(generate())),
                    mimetype="text/event-stream", headers=headers)


@app.route("/", methods=["GET", "POST"])
def index():
    response_message = None
    if request.method == "POST":
        global latest_form_data

        # Retrieve form data from the request (no user auth fields)
        latest_form_data = {
            "question": request.form.get("main_question", "N/A"),
            "agent1_system": request.form.get("agent1_system", ""),
            "agent1_voice": request.form.get("agent1_voice", ""),
            "agent1_custom": request.form.get("agent1_custom_voice", ""),
            "agent2_system": request.form.get("agent2_system", ""),
            "agent2_voice": request.form.get("agent2_voice", ""),
            "agent2_custom": request.form.get("agent2_custom_voice", ""),
            "agent3_system": request.form.get("agent3_system", ""),
            "agent3_voice": request.form.get("agent3_voice", ""),
            "agent3_custom": request.form.get("agent3_custom_voice", ""),
            "llm_model": request.form.get("llm_model", ""),
            "iterations": request.form.get("iterations", "1"),
        }

        # Generate a unique sessionID
        session_id = str(uuid.uuid4())
        sessions[session_id] = latest_form_data

        response_message = {
            "message": "Form data received. Streaming updates will begin shortly.",
            "session_id": session_id
        }

        # Return a short message so your AJAX can then open the SSE connection.
        return jsonify(response_message)

    return render_template("index.html", response_message=response_message)

@app.route('/api/healthz')
def health_check():
    return 'OK'

@app.route('/error')
def error_test():
    return render_template('error.html')

def compile_tex_to_pdf(tex_path: str | os.PathLike, out_dir: str | os.PathLike | None = None) -> str:
    """
    Compile a .tex file to a PDF and return the absolute path to the PDF.
    Uses latexmk (or pdflatex fallback if you prefer). Raises on failure.
    """
    tex_path = Path(tex_path).resolve()
    out_dir = Path(out_dir).resolve() if out_dir else tex_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # latexmk is robust; swap for pdflatex if needed.
    cmd = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        f"-outdir={str(out_dir)}",
        str(tex_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"latexmk failed (code {result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    pdf_path = out_dir / (tex_path.stem + ".pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected PDF not found: {pdf_path}")
    return str(pdf_path)

def generate_pdf(tex_path: str, use_secure: bool = False, as_attachment: bool = True):
    pdf_path = compile_tex_to_pdf(tex_path)
    if has_request_context():
        return send_file(pdf_path, as_attachment=as_attachment,
                         download_name=os.path.basename(pdf_path))
    else:
        return pdf_path

@app.route('/generate_pdf/<path:tex_relpath>', methods=['GET'])
def generate_pdf_route(tex_relpath):
    # Resolve the .tex path inside a safe base directory
    base_dir = Path(__file__).resolve().parent / "generated_tex"
    tex_path = (base_dir / tex_relpath).resolve()

    # Prevent path traversal outside base_dir
    if base_dir not in tex_path.parents and tex_path != base_dir:
        return jsonify({"error": "Invalid path"}), 400

    try:
        pdf_path = compile_tex_to_pdf(tex_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Serve the PDF to the browser
    return send_file(
        pdf_path,
        as_attachment=True,  # or False if you want inline
        download_name=os.path.basename(pdf_path)
    )

        
if __name__ == '__main__':
    app.run(debug=True)
