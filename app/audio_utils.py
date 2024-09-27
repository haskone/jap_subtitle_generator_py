import os
import logging

from moviepy.editor import VideoFileClip
from pydub import AudioSegment

import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

TEMP_AUDIO_CODEC = "pcm_s16le"
TEMP_AUDIO_FORMAT = "wav"


def extract_audio(
    videofile_path,
    audio_file_path,
    target_sample_rate=16000,
    output_format="wav",
    start_time=None,
    end_time=None,
    normalize_audio=False,
):
    try:
        with VideoFileClip(videofile_path) as video:
            audio = video.audio

            if start_time is not None and end_time is not None:
                logger.info(f"Extracting audio from {start_time} to {end_time}")
                audio = audio.subclip(start_time, end_time)

            temp_audio_path = f"{audio_file_path}.temp.{TEMP_AUDIO_FORMAT}"
            logger.info(f"Writing temporary audio to {temp_audio_path}")
            audio.write_audiofile(
                temp_audio_path, codec=TEMP_AUDIO_CODEC,
            )

        # Resample audio here
        logger.info(f"Resampled audio to {target_sample_rate} Hz")
        y, _ = librosa.load(temp_audio_path, sr=target_sample_rate)

        # May need to normalize audio in some cases?
        # Need to play with more files to see if this is necessary.
        # I guess for most official anime audio, this is not necessary.
        if normalize_audio:
            logger.info("Normalizing audio")
            y = librosa.util.normalize(y)

        logger.info(f"Saving audio to {audio_file_path}")
        sf.write(audio_file_path, y, target_sample_rate, format=output_format)

        logger.info(f"Removing temporary audio file: {temp_audio_path}")
        # That *.temp.wav
        os.remove(temp_audio_path)

        logger.info(f"Saved audio file to {audio_file_path}")
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise


def get_audio_duration(audio_file_path) -> float:
    """Get the duration of an audio file in seconds."""
    logger.info(f"Getting audio duration of {audio_file_path}")
    audio = AudioSegment.from_wav(audio_file_path)
    return len(audio) / 1000.0  # pydub works in milliseconds


def split_audio(audio_file_path, chunk_file_path, chunk_duration=60, overlap=5) -> list[str]:
    """Split audio file into overlapping chunks."""
    logger.info(f"Splitting audio file {audio_file_path} into chunks")
    audio = AudioSegment.from_wav(audio_file_path)
    duration = len(audio)
    chunk_length = chunk_duration * 1000  # pydub works in milliseconds
    overlap_length = overlap * 1000

    chunks = []
    for start in range(0, duration, chunk_length - overlap_length):
        end = start + chunk_length
        chunk: AudioSegment = audio[start:end]
        chunk_file = f"{audio_file_path}_chunk_{start//1000}-{end//1000}.wav"
        chunk_file_path = os.path.join(chunk_file_path, chunk_file)
        chunk.export(chunk_file_path, format="wav")
        chunks.append(chunk_file_path)

    return chunks
