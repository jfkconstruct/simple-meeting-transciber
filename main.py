
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
        .output { white-space: pre-wrap; background: #fff; padding: 15px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Video Processing Tool</h1>
        <form action="/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".mp4,.avi,.mov,.wav" required>
            <input type="submit" value="Process Video">
        </form>
        {% if output_path %}
        <div class="output">
            <h2>Processing Complete!</h2>
            <p>Your file has been processed. <a href="/download/{{ output_filename }}">Download Results</a></p>
        </div>
        {% endif %}
    </div>
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

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('output', filename),
                    as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
