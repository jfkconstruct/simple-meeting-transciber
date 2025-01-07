import os

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

# Set up OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']


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
    output_path = generate_output_file(file_path, transcript, summary,
                                       next_steps)

    print(f"Output file generated: {output_path}")

    # Clean up temporary audio chunks
    for chunk in audio_chunks:
        os.remove(chunk)


def main():
    upload_folder = 'uploads'
    output_folder = 'output'

    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    print("Please place your audio or video file in the 'uploads' folder.")
    input("Press Enter when you've uploaded the file...")

    files = os.listdir(upload_folder)
    if not files:
        print(
            "No files found in the 'uploads' folder. Thank you for using our service. Goodbye!"
        )
        return

    for filename in files:
        file_path = os.path.join(upload_folder, filename)
        process_file(file_path)


if __name__ == "__main__":
    main()
