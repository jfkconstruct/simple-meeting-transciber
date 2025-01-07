# Simple Meeting Transcription Tool

This Repl provides a tool for transcribing video or audio files and generating insightful summaries and next steps. It's designed to help users quickly extract valuable information from media files.

## Features

- Supports various video and audio formats (mp4, avi, mov, wav)
- Transcribes speech to text
- Generates a concise summary of the transcript
- Suggests next steps based on the content
- Outputs results to a markdown file

## Configuration

This Repl is pre-configured and ready to use. However, you'll need to set up your OpenAI API key for the summary and next steps generation.

## Setup

1. Clone this Repl to your account.
2. Add your OpenAI API key to the Replit Secrets:
   - Click on the padlock icon in the sidebar to open the Secrets tab.
   - Create a new secret with the key `OPENAI_APIKEY` and your API key as the value.
3. Run the Repl.

## Usage

1. When you run the Repl, it will create two folders: `uploads` and `output`.
2. Place your video or audio file in the `uploads` folder.
3. Press Enter in the console when prompted.
4. The script will process your file and generate an output text file in the `output` folder.

## Output

The generated output file will contain:
- Full transcript of the audio
- A summary of the content
- Suggested next steps based on the content

## Dependencies

This project uses several Python libraries:
- `openai` for generating summaries and next steps
- `moviepy` for handling video files
- `speech_recognition` for transcribing audio to text

These dependencies are automatically installed when you run the Repl.

## Customization

If you want to modify the AI models used for transcription, summary, or next steps generation:
1. Open the `utils.py` file.
2. In the `transcribe_audio` function, you can change the `model` parameter to use a different Whisper model.
3. In the `generate_summary` and `generate_next_steps` functions, you can modify the `model` parameter to use a different OpenAI model.

## Note

Please ensure you have the necessary rights to transcribe and analyze the audio/video files you upload.

## More

For more templates and guides, check out https://replit.com/templates or https://replit.com/guides.

Happy transcribing and analyzing!
