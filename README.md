# Simple Meeting Transcription Tool

A tool for transcribing video or audio files and generating insightful summaries and next steps. It's designed to help users quickly extract valuable information from media files.

## Features

- Supports various video and audio formats (mp4, avi, mov, wav)
- Fast and accurate transcription using faster-whisper
- Automatic GPU acceleration with CPU fallback
- Voice Activity Detection (VAD) to filter out silence
- Word-level timestamps for precise navigation
- Local transcription processing (no API needed for transcription)
- Generates a concise summary of the transcript using OpenAI
- Suggests next steps based on the content
- Outputs results to a markdown file
- Web interface for easy file upload and processing

## Requirements

- Python 3.9 or higher
- CUDA 12.x and cuDNN 9.x (optional, for GPU acceleration)
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

4. (Optional) For GPU acceleration:
   - Install CUDA 12.x from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Install cuDNN 9.x from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - The system will automatically use GPU if available, otherwise fall back to CPU

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
- Full transcript of the audio with timestamps
- A summary of the content
- Suggested next steps based on the content

## Performance Notes

- GPU acceleration (if available) provides significantly faster processing
- CPU mode uses int8 quantization for optimal performance
- Voice Activity Detection (VAD) automatically removes silence
- Processing time depends on:
  - File length
  - Available hardware (GPU/CPU)
  - Amount of silence in the audio

## Customization

The transcription process can be customized by modifying parameters in `utils.py`:
1. Model size: The default model is `large-v3` for best accuracy
2. VAD parameters: Adjust silence detection sensitivity
3. Beam size: Controls the trade-off between speed and accuracy (default: 5)

## Note

Please ensure you have the necessary rights to transcribe and analyze the audio/video files you upload.

Happy transcribing and analyzing!
