import sys
import threading
import queue
import time
import os
import subprocess
import json
import uuid
from dotenv import load_dotenv
from flask import Flask, Response, render_template, request, stream_with_context, jsonify, send_file
from werkzeug.utils import secure_filename
from src.goldilocks_and_the_3_agents import run_this_thing

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

@app.route('/generate_pdf/<path:tex_filename>', methods=['GET'])
def generate_pdf(tex_filename, use_secure=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "generated_tex")

    # Strict extension check
    if os.path.splitext(tex_filename)[1].lower() != '.tex':
        return "Invalid file type. Only .tex files are allowed.", 400

    # Sanitize name
    safe_tex_filename = os.path.basename(secure_filename(tex_filename)) if use_secure else os.path.basename(tex_filename)

    tex_path = os.path.join(output_dir, safe_tex_filename)
    pdf_path = tex_path.replace('.tex', '.pdf')

    if not os.path.exists(tex_path):
        return f"TeX file not found", 404

    os.makedirs(output_dir, exist_ok=True)

    for i in range(2):
        try:
            subprocess.run(
                [
                    'xelatex',
                    '-interaction=nonstopmode',
                    '-halt-on-error',
                    '-no-shell-escape',
                    f'-output-directory={output_dir}',
                    tex_path
                ],
                check=True,
                timeout=30,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            # Log e.stdout/e.stderr somewhere safe if you have logging
            return "Error generating PDF.", 500
        except subprocess.TimeoutExpired:
            return "PDF generation timed out.", 504

    if not os.path.exists(pdf_path):
        return "PDF not generated", 500

    return send_file(
        pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=os.path.basename(pdf_path)
    )

if __name__ == '__main__':
    app.run(debug=True)
