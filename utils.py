import os

import openai
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def convert_to_audio(file_path):
    """Convert video to audio if necessary."""
    if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        video = VideoFileClip(file_path)
        audio = video.audio
        audio_path = file_path.rsplit('.', 1)[0] + '.wav'
        audio.write_audiofile(audio_path) if audio else None
        return audio_path
    return file_path


def chunk_audio(audio_path, chunk_length_ms=60000):
    """Split audio into chunks of a specified length."""
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"{audio_path[:-4]}_chunk_{i//chunk_length_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def transcribe_audio(audio_chunks):
    """Transcribe audio chunks using OpenAI Whisper."""
    full_transcript = ""
    with tqdm(total=len(audio_chunks), desc="Transcribing",
              unit="chunk") as pbar:
        for chunk in audio_chunks:
            with open(chunk, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1", file=audio_file)
            full_transcript += transcript.text + " "
            pbar.update(1)
    return full_transcript.strip()


def chunk_text(text, chunk_size=127000):
    """Split text into chunks."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    for word in words:
        if current_size + len(word) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def generate_summary(transcript_chunks, max_tokens=4000):
    """Generate a summary for long transcripts."""
    chunk_summaries = []
    for chunk in transcript_chunks:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role":
                "system",
                "content":
                "You are a helpful assistant that summarizes text."
            }, {
                "role":
                "user",
                "content":
                f"Please provide a brief summary of the following transcript chunk:\n\n{chunk}"
            }],
            max_tokens=max_tokens)
        chunk_summaries.append(response.choices[0].message.content)

    # Combine chunk summaries
    combined_summary = " ".join(chunk_summaries)

    # Generate final summary
    final_response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role":
            "system",
            "content":
            "You are a helpful assistant that creates concise summaries."
        }, {
            "role":
            "user",
            "content":
            f"Please provide a final, concise summary based on these chunk summaries:\n\n{combined_summary}"
        }],
        max_tokens=max_tokens)

    return final_response.choices[0].message.content


def generate_next_steps(transcript_chunks, max_tokens=4000):
    """Generate next steps for long transcripts."""
    chunk_next_steps = []
    for chunk in transcript_chunks:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role":
                "system",
                "content":
                "You are a helpful assistant that suggests next steps based on a transcript."
            }, {
                "role":
                "user",
                "content":
                f"Based on the following transcript chunk, what are the next steps?\n\n{chunk}"
            }],
            max_tokens=max_tokens)
        chunk_next_steps.append(response.choices[0].message.content)

    # Combine chunk next steps
    combined_next_steps = " ".join(chunk_next_steps)

    # Generate final next steps
    final_response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role":
            "system",
            "content":
            "You are a helpful assistant that creates concise action plans."
        }, {
            "role":
            "user",
            "content":
            f"Please provide a final, concise set of next steps based on these suggestions:\n\n{combined_next_steps}"
        }],
        max_tokens=max_tokens)

    return final_response.choices[0].message.content


def generate_output_file(file_path, transcript, summary, next_steps):
    """Generate a formatted Markdown output file."""
    output_path = os.path.join(
        'output',
        os.path.basename(file_path).rsplit('.', 1)[0] + '_output.md')

    with open(output_path, 'w') as f:
        f.write(f"# Meeting Notes: {os.path.basename(file_path)}\n\n")
        f.write(f"## Summary\n\n")
        f.write(summary + "\n\n")
        f.write(f"## Next Steps\n\n")
        f.write(next_steps + "\n\n")
        f.write(f"## Full Transcript\n\n")
        f.write(transcript + "\n\n")

    return output_path
