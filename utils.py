import os
import sys
import time
import logging
import speech_recognition as sr
import openai
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
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

def chunk_audio(audio_path, chunk_length_ms=120000):  # 2 minutes per chunk
    """Split audio into chunks of a specified length with size limit handling."""
    logger.info(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    
    # First convert to mono and standardize sample rate to reduce size
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        
        # Compress chunk if needed
        chunk = compress_audio(chunk)
        
        # Export with reduced quality if still too large
        chunk_path = f"{audio_path[:-4]}_chunk_{i//chunk_length_ms}.wav"
        export_kwargs = {
            'format': 'wav',
            'parameters': ['-ar', '16000', '-ac', '1']
        }
        
        chunk.export(chunk_path, **export_kwargs)
        
        # Verify final size
        size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        logger.info(f"Chunk {i//chunk_length_ms} size: {size_mb:.2f}MB")
        
        if size_mb > 25:
            logger.warning(f"Chunk {chunk_path} still too large. Reducing quality further.")
            # If still too large, reduce quality more aggressively
            export_kwargs['parameters'].extend(['-q:a', '3'])
            chunk.export(chunk_path, **export_kwargs)
            size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            logger.info(f"Reduced chunk {i//chunk_length_ms} size: {size_mb:.2f}MB")
        
        chunks.append(chunk_path)
    
    return chunks

def convert_to_audio(file_path):
    """Convert video to audio if necessary."""
    if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        video = VideoFileClip(file_path)
        audio = video.audio
        audio_path = file_path.rsplit('.', 1)[0] + '.wav'
        audio.write_audiofile(audio_path) if audio else None
        video.close()
        return audio_path
    return file_path

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def transcribe_chunk(chunk):
    """Transcribe a single audio chunk with retry logic."""
    try:
        logger.info(f"Starting transcription of chunk: {chunk}")
        start_time = time.time()
        
        # Ensure the file exists and is readable
        if not os.path.exists(chunk):
            raise FileNotFoundError(f"Chunk file not found: {chunk}")
            
        file_size = os.path.getsize(chunk) / (1024 * 1024)  # Size in MB
        logger.info(f"Processing chunk {chunk} (Size: {file_size:.2f}MB)")
        
        with open(chunk, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                timeout=60  # Increased timeout
            )
        
        duration = time.time() - start_time
        logger.info(f"Finished transcribing chunk {chunk} in {duration:.2f} seconds")
        
        # Clean up the chunk file after successful transcription
        try:
            os.remove(chunk)
            logger.info(f"Cleaned up chunk file: {chunk}")
        except Exception as e:
            logger.warning(f"Failed to clean up chunk {chunk}: {e}")
            
        return transcript.text
    except Exception as e:
        logger.error(f"Error transcribing chunk {chunk}: {str(e)}")
        raise  # Let the retry decorator handle it

def transcribe_audio(audio_chunks):
    """Transcribe audio chunks using OpenAI Whisper with parallel processing."""
    full_transcript = []
    failed_chunks = []
    
    logger.info(f"Starting transcription of {len(audio_chunks)} chunks")
    
    # Process chunks in smaller batches to avoid overwhelming the API
    batch_size = 2
    for i in range(0, len(audio_chunks), batch_size):
        batch = audio_chunks[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} of {(len(audio_chunks) + batch_size - 1)//batch_size}")
        
        with ThreadPoolExecutor(max_workers=1) as executor:  # Process one chunk at a time
            futures = {executor.submit(transcribe_chunk, chunk): chunk for chunk in batch}
            
            with tqdm(total=len(batch), desc=f"Batch {i//batch_size + 1}", unit="chunk") as pbar:
                for future in as_completed(futures, timeout=600):  # 10 minute timeout per batch
                    chunk = futures[future]
                    try:
                        result = future.result()
                        full_transcript.append(result)
                        logger.info(f"Successfully processed chunk: {chunk}")
                    except Exception as e:
                        error_msg = f"Failed to process chunk {chunk}: {str(e)}"
                        logger.error(error_msg)
                        full_transcript.append(f"[{error_msg}]")
                        failed_chunks.append(chunk)
                    finally:
                        pbar.update(1)
        
        # Add a small delay between batches
        if i + batch_size < len(audio_chunks):
            logger.info("Waiting 5 seconds before processing next batch...")
            time.sleep(5)
    
    if failed_chunks:
        logger.warning(f"Failed to process {len(failed_chunks)} chunks: {failed_chunks}")
    
    return " ".join(full_transcript).strip()

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
