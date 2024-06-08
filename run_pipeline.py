import os
import time

from sub_gen.create_audio_file_reduced import extract_audio
from sub_gen.translation_model import translate_one
from sub_gen.whisper_model import transcribe
from dotenv import load_dotenv

load_dotenv()
# Assuming you have a .env file in the same directory as this script
# and filename in "data" dir of the project
FILENAME = os.getenv("VIDEO_FILENAME")

DEFAULT_SHOWING_TIME = 5
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(basedir, "data")
filename = os.path.join(datadir, FILENAME)

audio_file_raw_path = os.path.join(datadir, f"{filename}_audio_raw.wav")
audio_file_reduced_path = os.path.join(datadir, f"{filename}_audio_reduced.wav")

# Start time track the total time of the process
# including audio extraction, audio cutting, and transcription
total_start_time = time.time()

extract_audio(filename, audio_file_raw_path, audio_file_reduced_path)
print("Audio extracted...")

##  Cut the audio into smaller chunks, in case you may need it:
# from sub_gen.audio_cut import cut_by_seconds
# sec_interval = 0.25 * 60 * 10 # 5 or 30? 600? The entire file seems too resource intensive
# output_wav_path = os.path.join(os.path.join(datadir, f"audio_cut_{sec_interval}_sec"), "item_" )
# files = cut_by_seconds(audio_file_reduced_path, output_wav_path, sec_interval)

files = [audio_file_reduced_path]  # Use just one file for now
print("Audio cut is done...")

sub_file_path = filename.split(".")[0] + ".sub"
with open(sub_file_path, "a", encoding="utf-8") as sub_file:
    for file_chunk_index, file in enumerate(files):
        print("Processing file:", file, "...")
        result = transcribe(file)
        print(result)

        print("Creating subtitles...")
        for idx, chunk in enumerate(result["chunks"], start=1):
            if (
                "timestamp" not in chunk
                or "text" not in chunk
                or not len(chunk["timestamp"])
            ):
                continue

            # start_time = chunk['timestamp'][0] + sec_interval * file_chunk_index
            start_time = chunk["timestamp"][0] + file_chunk_index
            if not chunk["timestamp"][1]:
                end_time = start_time + DEFAULT_SHOWING_TIME
            else:
                # end_time = chunk['timestamp'][1] + sec_interval * file_chunk_index
                end_time = chunk["timestamp"][1] + file_chunk_index

            start_time_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
            end_time_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"

            sub_file.write(f"{idx + file_chunk_index}\n")
            sub_file.write(f"{start_time_srt} --> {end_time_srt}\n")
            sub_file.write(f'{translate_one(chunk["text"])}\n\n')


total_end_time = time.time()
print(f"Total runtime is {total_end_time - total_start_time:.2f} seconds")
