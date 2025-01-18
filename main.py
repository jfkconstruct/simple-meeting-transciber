
import os
import time
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename

import openai
from utils import (
    chunk_audio,
    chunk_text,
    convert_to_audio,
    generate_next_steps,
    generate_output_file,
    generate_summary,
    transcribe_audio,
)

app = Flask(__name__)
openai.api_key = os.environ['OPENAI_API_KEY']

# HTML template for the upload interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Video Processing Tool</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .button { 
            display: inline-block;
            padding: 10px 20px;
            background: #659cef;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }
        .button:hover {
            background: #5080d0;
        }
        .output { white-space: pre-wrap; background: #fff; padding: 15px; margin-top: 20px; }
        .progress-container { display: none; margin-top: 20px; }
        .progress-bar { 
            width: 100%;
            background-color: #f0f0f0;
            padding: 3px;
            border-radius: 3px;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, .2);
        }
        .progress-bar-fill {
            display: block;
            height: 22px;
            background-color: #659cef;
            border-radius: 3px;
            transition: width 500ms ease-in-out;
            width: 0%;
        }
        .status-text {
            margin-top: 5px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Video Processing Tool</h1>
        <form id="uploadForm" action="/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".mp4,.avi,.mov,.wav" required>
            <input type="submit" value="Process Video">
        </form>
        <div id="progressContainer" class="progress-container">
            <div class="progress-bar">
                <span class="progress-bar-fill"></span>
            </div>
            <div id="statusText" class="status-text">Processing...</div>
        </div>
        {% if output_path %}
        <div class="output">
            <h2>Processing Complete!</h2>
            <p>Your file has been processed.</p>
            <div style="margin-top: 15px;">
                <a href="/summary/{{ output_filename }}" class="button" style="margin-right: 10px;">View Summary</a>
                <a href="/download/{{ output_filename }}" class="button">Download Full Results</a>
            </div>
        </div>
        {% endif %}
    </div>
    <script>
        document.getElementById('uploadForm').onsubmit = function() {
            document.getElementById('progressContainer').style.display = 'block';
            const progressBar = document.querySelector('.progress-bar-fill');
            const statusText = document.getElementById('statusText');
            let progress = 0;
            
            // Simulate progress for long-running process
            const interval = setInterval(() => {
                if (progress < 90) {
                    progress += Math.random() * 10;
                    progressBar.style.width = progress + '%';
                    if (progress < 30) {
                        statusText.textContent = 'Converting audio...';
                    } else if (progress < 60) {
                        statusText.textContent = 'Transcribing...';
                    } else {
                        statusText.textContent = 'Generating summary...';
                    }
                }
            }, 1000);
            
            return true;
        };
    </script>
</body>
</html>
'''

def process_file(file_path):
    """Process the uploaded file and generate the document."""
    print("> Converting audio")
    audio_path = convert_to_audio(file_path)
    audio_chunks = chunk_audio(audio_path)
    print("> Transcribing audio")
    transcript = transcribe_audio(audio_chunks)
    transcript_chunks = chunk_text(transcript)
    summary = generate_summary(transcript_chunks)
    next_steps = generate_next_steps(transcript_chunks)

    print("> Generating output")
    output_path = generate_output_file(file_path, transcript, summary, next_steps)

    # Clean up temporary audio chunks
    for chunk in audio_chunks:
        os.remove(chunk)
        
    return output_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
            
        filename = secure_filename(file.filename)
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        output_path = process_file(file_path)
        output_filename = os.path.basename(output_path)
        
        return render_template_string(HTML_TEMPLATE, 
                                   output_path=output_path,
                                   output_filename=output_filename)
    
    return render_template_string(HTML_TEMPLATE)

@app.route('/summary/<filename>')
def view_summary(filename):
    file_path = os.path.join('output', filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
            # Extract summary section
            start = content.find('## Summary')
            end = content.find('##', start + 1)
            summary = content[start:end] if end != -1 else content[start:]
            return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Summary</title>
                    <style>
                        body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
                        .container { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                        .button { 
                            display: inline-block;
                            padding: 10px 20px;
                            background: #659cef;
                            color: white;
                            text-decoration: none;
                            border-radius: 5px;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div style="white-space: pre-wrap;">{{ summary }}</div>
                        <div style="margin-top: 20px;">
                            <a href="/download/{{ filename }}" class="button">Download Full Results</a>
                        </div>
                    </div>
                </body>
                </html>
            ''', summary=summary, filename=filename)
    return 'File not found', 404

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('output', filename),
                    as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
