import os

from sub_gen.create_audio_file_reduced import extract_audio
from sub_gen.audio_cut import cut_by_minutes
from sub_gen.translation_model import translate_one
from sub_gen.whisper_model import transcribe

DEFAULT_SHOWING_TIME = 5
sec_interval = 60 * 10 # 5 or 30? 600? The entire file seems too resource intensive
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(basedir, 'data')
filename = os.path.join(datadir, "<filename>.mkv")

audio_file_raw_path = os.path.join(datadir, f"{filename}_audio_raw.wav")
audio_file_reduced_path = os.path.join(datadir, f"{filename}_audio_reduced.wav")

output_wav_path = os.path.join(os.path.join(datadir, f"audio_cut_{sec_interval}_sec"), "item_" )

# extract_audio(filename, audio_file_raw_path, audio_file_reduced_path)
print('Audio extracted...')

# files = cut_by_minutes(audio_file_reduced_path, output_wav_path, sec_interval)
files = [audio_file_reduced_path]
print('Audio cut is done...')

sub_file_path = filename.split('.')[0] + '.sub'
with open(sub_file_path, 'a', encoding='utf-8') as sub_file:
    for file_chunk_index, file in enumerate(files):
        print('Processing file:', file, '...')
        result = transcribe(file)
        print(result)

        for idx, chunk in enumerate(result['chunks'], start=1):
            if 'timestamp' not in chunk or 'text' not in chunk or not len(chunk['timestamp']):
                continue

            start_time = chunk['timestamp'][0] + sec_interval * file_chunk_index
            if not chunk['timestamp'][1]:
                end_time = start_time + DEFAULT_SHOWING_TIME
            else:
                end_time = chunk['timestamp'][1] + sec_interval * file_chunk_index

            # Convert times to SUB format times
            start_time_srt = f'{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}'
            end_time_srt = f'{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}'

            # Write to SRT file
            sub_file.write(f'{idx + file_chunk_index}\n')
            sub_file.write(f'{start_time_srt} --> {end_time_srt}\n')
            sub_file.write(f'{translate_one(chunk["text"])}\n\n')
