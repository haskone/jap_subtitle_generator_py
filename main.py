import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor

from app.audio_utils import (
    extract_audio,
    split_audio,
    get_audio_duration,
)
from app.translation_model import translate_batch
from app.whisper_model import transcribe

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from constants import (
    DEFAULT_SHOWING_TIME,
    CHUNK_SIZE,
    TRANSLATION_BATCH_SIZE,
    ENV_CHUNK_SUBDIR,
    ENV_DATA_DIR,
    ENV_SUB_EXT,
    ENV_VIDEO_FILENAME,
)


CHUNK_DIR = os.getenv(ENV_CHUNK_SUBDIR)
DATA_DIR = os.getenv(ENV_DATA_DIR)
FILENAME = os.getenv(ENV_VIDEO_FILENAME)
SUB_EXT = os.getenv(ENV_SUB_EXT)

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(basedir, DATA_DIR)
filename = os.path.join(datadir, FILENAME)

audio_file_path = os.path.join(datadir, f"{filename}_audio.wav")
chunk_file_path = os.path.join(datadir, CHUNK_DIR)


def print_envs():
    logger.info(f"Basedir: {basedir}")
    logger.info(f"Datadir: {datadir}")
    logger.info(f"Filename: {filename}")
    logger.info(f"Audio file path: {audio_file_path}")
    logger.info(f"Chunk folder path: {chunk_file_path}")


def process_chunk(chunk: dict, file_chunk_index: int) -> list[str]:
    segments = []
    texts_to_translate = []

    for idx, segment in enumerate(chunk["chunks"], start=1):
        if (
            "timestamp" not in segment
            or "text" not in segment
            or not segment["timestamp"]
        ):
            continue

        start_time = segment["timestamp"][0] + file_chunk_index * CHUNK_SIZE
        end_time = (
            segment["timestamp"][1] + file_chunk_index * CHUNK_SIZE
            if segment["timestamp"][1]
            else start_time + DEFAULT_SHOWING_TIME
        )

        start_time_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
        end_time_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"

        segments.append((idx + file_chunk_index, start_time_srt, end_time_srt))
        texts_to_translate.append(segment["text"])

    translated_texts = translate_batch(
        texts_to_translate, batch_size=TRANSLATION_BATCH_SIZE
    )

    results = [
        f"{idx}\n{start} --> {end}\n{text}\n\n"
        for (idx, start, end), text in zip(segments, translated_texts)
    ]

    return results


def main(split_into_chunks=False):
    total_start_time = time.time()

    try:
        extract_audio(filename, audio_file_path, normalize_audio=True)
        logger.info("Audio extracted...")

        audio_duration = get_audio_duration(audio_file_path)
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")

        with ThreadPoolExecutor() as executor:
            if split_into_chunks:
                audio_chunks = split_audio(
                    audio_file_path=audio_file_path,
                    chunk_file_path=chunk_file_path,
                )
                future_transcriptions = [
                    executor.submit(transcribe, chunk) for chunk in audio_chunks
                ]
            else:
                future_transcriptions = [executor.submit(transcribe, audio_file_path)]

            sub_file_path = f"{filename.split('.')[0]}.{SUB_EXT}"
            with open(sub_file_path, "w", encoding="utf-8") as sub_file:
                for file_chunk_index, future in enumerate(future_transcriptions):
                    chunk_result = future.result()
                    logger.info(f"Processing chunk {file_chunk_index + 1}...")

                    results = process_chunk(chunk_result, file_chunk_index)
                    sub_file.writelines(results)

        total_end_time = time.time()
        logger.info(f"Total runtime is {total_end_time - total_start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    print_envs()
    main(split_into_chunks=True)
