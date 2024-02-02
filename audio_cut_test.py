from pydub import AudioSegment

def cut_first_10_minutes(input_file_path, output_file_path):
    # Load the audio file
    audio = AudioSegment.from_wav(input_file_path)

    # Calculate 10 minutes in milliseconds

    left_right = 0.5 * 60 * 1000
    time_right = 0.2 * 60 * 1000


    # Cut the first 10 minutes
    first_10_minutes = audio[left_right:left_right + time_right]

    # Export the first 10 minutes to a new file
    first_10_minutes.export(output_file_path, format="wav")

# Specify your input WAV file and the output file name
input_wav_path = "audio_reduced.wav"
output_wav_path = "audio_reduced_item.wav"

# Cut the first 10 minutes of the WAV file
cut_first_10_minutes(input_wav_path, output_wav_path)
