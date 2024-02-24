from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf


def extract_audio(
    videofile_path,
    audio_file_raw_path,
    audio_file_reduced_path,
    target_sample_rate=16000,
    output_format="wav",
):
    video_clip = VideoFileClip(videofile_path)
    audio = video_clip.audio

    audio.write_audiofile(audio_file_raw_path, codec="pcm_s16le")

    audio, _ = librosa.load(audio_file_raw_path, sr=target_sample_rate)
    sf.write(audio_file_reduced_path, audio, target_sample_rate, format=output_format)

    print("Saved audio file to", audio_file_reduced_path)
