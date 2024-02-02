import librosa
import soundfile as sf

def reduce_audio_file_size(input_file_path, output_file_path, target_sample_rate=16000, output_format='wav'):
    # Load the audio file with librosa, automatically resampling to the target sample rate
    audio, _ = librosa.load(input_file_path, sr=target_sample_rate)

    # Save the resampled audio to the desired output format
    # Note: soundfile does not support saving to mp3, this line is for demonstration of the concept.
    # For MP3 saving, consider using pydub or adjust format to 'wav' or another supported format by soundfile.
    sf.write(output_file_path, audio, target_sample_rate, format=output_format)

# Define your input WAV file and output file name
input_wav_path = './audio.wav'
output_path = './audio_reduced.wav'  # Change to 'reduced_audio_file.wav' if using soundfile

# Call the function to reduce the file size
reduce_audio_file_size(input_wav_path, output_path)

# Note: This code snippet uses soundfile for demonstration, which does not support MP3.
# For actual MP3 support, consider using pydub as follows:
# from pydub import AudioSegment
# sound = AudioSegment.from_wav(input_wav_path)
# sound = sound.set_frame_rate(target_sample_rate)
# sound.export(output_path, format="mp3", bitrate="128k")
