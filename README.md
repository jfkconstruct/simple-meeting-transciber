# Simple Meeting Transcription Tool

A tool for transcribing video or audio files and generating insightful summaries and next steps. It's designed to help users quickly extract valuable information from media files.

## Features

- Supports various video and audio formats (mp4, avi, mov, wav)
- Fast and accurate transcription using faster-whisper
- Local transcription processing (no API needed for transcription)
- Generates a concise summary of the transcript using OpenAI
- Suggests next steps based on the content
- Outputs results to a markdown file
- Web interface for easy file upload and processing

## Requirements

- Python 3.9 or higher
- CUDA 12 and cuDNN 8 (optional, for GPU acceleration)
- OpenAI API key (for summary generation only)

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/jfkconstruct/simple-meeting-transciber.git
   cd simple-meeting-transciber
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the web server:
   ```bash
   python main.py
   ```

2. Open your browser and navigate to `http://localhost:3000`

3. Upload your video or audio file through the web interface

4. The script will process your file and generate an output text file in the `output` folder

## Output

The generated output file will contain:
- Full transcript of the audio
- A summary of the content
- Suggested next steps based on the content

## Dependencies

This project uses several Python libraries:
- `faster-whisper` for local transcription processing
- `openai` for generating summaries and next steps
- `moviepy` for handling video files
- `flask` for the web interface

## Customization

The transcription process can be customized by modifying parameters in `utils.py`:
1. Model size: The default model is `large-v3` for best accuracy, but you can change to other sizes like `small` or `medium` for faster processing
2. Batch size: Adjust based on your GPU memory
3. Beam size: Controls the trade-off between speed and accuracy

## Note

Please ensure you have the necessary rights to transcribe and analyze the audio/video files you upload.

## Performance Notes

- GPU acceleration is recommended for faster processing
- CPU-only mode is supported but will be significantly slower
- Processing time depends on the file length and chosen model size

Happy transcribing and analyzing!
