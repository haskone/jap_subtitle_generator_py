from pydub import AudioSegment


def cut_by_seconds(input_file_path, output_file_path_base, sec_interval):
    audio = AudioSegment.from_wav(input_file_path)
    # AudioSegments are slicable using milliseconds. for example:
    # a = AudioSegment.from_mp3(mp3file)
    # first_second = a[:1000] # get the first second of an mp3
    # slice = a[5000:10000] # get a slice from 5 to 10 seconds of an mp3

    chunk_files = []
    multiplier = sec_interval * 1000
    for sec in range(0, len(audio) // multiplier):
        curr_sec = sec * multiplier
        next_sec = audio[curr_sec : curr_sec + multiplier]

        chunk_filename = output_file_path_base + f"{sec}.wav"
        next_sec.export(chunk_filename, format="wav")
        chunk_files.append(chunk_filename)

    print(f"saved {len(audio) // multiplier} chunks\n")
    return chunk_files
