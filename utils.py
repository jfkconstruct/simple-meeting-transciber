import os
import sys
import time
import logging
import speech_recognition as sr
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24MB to stay safely under OpenAI's 25MB limit

def compress_audio(audio_segment, target_size_mb=20):
    """Compress audio to target size while maintaining quality."""
    original_size = len(audio_segment.raw_data) / (1024 * 1024)
    logger.info(f"Original audio size: {original_size:.2f}MB")
    
    # If already under target size, return as is
    if original_size <= target_size_mb:
        return audio_segment
    
    # Convert to mono if stereo
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)
    
    # Reduce sample rate if needed (preserve quality better than bitrate reduction)
    if len(audio_segment.raw_data) / (1024 * 1024) > target_size_mb:
        audio_segment = audio_segment.set_frame_rate(16000)  # 16kHz is good for speech
    
    return audio_segment

def convert_to_audio(file_path):
    """Convert video to audio if necessary."""
    if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        logger.info(f"Converting video file to audio: {file_path}")
        video = VideoFileClip(file_path)
        audio = video.audio
        audio_path = file_path.rsplit('.', 1)[0] + '.wav'
        audio.write_audiofile(audio_path) if audio else None
        video.close()
        return audio_path, True  # Return flag indicating if conversion happened
    return file_path, False

def transcribe_audio(audio_path):
    """Transcribe audio using faster-whisper."""
    logger.info("Initializing faster-whisper model...")
    
    # Initialize model with float16 on GPU if available, otherwise use int8 on CPU
    try:
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        logger.info("Using GPU with float16")
    except (ImportError, RuntimeError):
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        logger.info("Using CPU with int8")
    
    logger.info("Starting transcription...")
    
    # Perform transcription with VAD filter and word timestamps
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,  # Filter out silence
        vad_parameters=dict(min_silence_duration_ms=500)  # Adjust silence threshold
    )
    
    # Convert generator to list to start transcription
    segments = list(segments)
    
    # Log detected language
    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    # Combine all segments into full transcript
    transcript = " ".join(segment.text for segment in segments)
    
    return transcript.strip()

def cleanup_temp_files(audio_path, was_converted):
    """Clean up temporary audio files from video conversion."""
    if was_converted:
        try:
            os.remove(audio_path)
            logger.info(f"Cleaned up temporary audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {audio_path}: {e}")

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
            model="gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
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
            model="gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
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
