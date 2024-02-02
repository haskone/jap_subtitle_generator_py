import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
import librosa
import numpy as np
from moviepy.editor import VideoFileClip

# Initialize the processor using the pre-trained model's name
processor = Wav2Vec2Processor.from_pretrained("vumichien/wav2vec2-large-xlsr-japanese")

model = Wav2Vec2ForCTC.from_pretrained("vumichien/wav2vec2-large-xlsr-japanese")

# Function to extract audio from mp4 and resample
def extract_audio_resample(mp4_path, target_sr=16_000):
    # video_clip = VideoFileClip(mp4_path)
    # audio = video_clip.audio
    # temp_audio_path = "./audio_reduced_10_mins.wav"

    # # audio.write_audiofile(temp_audio_path, codec='pcm_s16le')

    # # Load the audio file
    # speech_array, sampling_rate = torchaudio.load(temp_audio_path)
    # # Resample the audio to the target sample rate
    # resampled_speech = librosa.resample(speech_array.numpy().squeeze(), orig_sr=sampling_rate, target_sr=target_sr)
    # # Convert the numpy array back to tensor

    # # resampled_speech_tensor = torch.tensor(resampled_speech).unsqueeze(0)  # Add channel dimension
    # # return resampled_speech_tensor, target_sr

    # return resampled_speech, target_sr

    audio_path = "./audio_reduced_item.wav"
    speech_array, sampling_rate = torchaudio.load(audio_path)
    # Since no resampling is needed, directly return the speech array and its sampling rate
    return speech_array.squeeze(), sampling_rate  # Remove the channel dimension if it exists


# Specify your MP4 file path here
mp4_file_path = "file.mp4"

# Extract and resample the audio from the MP4 file
speech_tensor, sr = extract_audio_resample(mp4_file_path)

# Process the audio for the model
inputs = processor(speech_tensor, sampling_rate=sr, return_tensors="pt", padding=True)


# ------------------ Generate subs ------------------ #
print('inputs', inputs, inputs.input_values.shape)

# Perform inference
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

print('logits', logits)

# # Decode the predicted ids to text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print('predicted_ids', predicted_ids)
print('transcription', transcription)

# Assuming `inputs` and `logits` are available from the previous steps
# and `sr` is the sampling rate of the audio

# Get the duration of the audio in seconds
audio_duration = speech_tensor.shape[-1] / sr

# Calculate the duration of each time step in the model output
time_step_duration = audio_duration / logits.shape[1]

# Placeholder: Simple method to estimate word timings (highly approximate)
# This example assumes uniform distribution of tokens, which is a rough approximation
word_timings = []
for i, token_id in enumerate(predicted_ids[0]):
    if token_id != processor.tokenizer.pad_token_id:  # Assuming non-pad tokens are relevant
        time_stamp = i * time_step_duration
        word_timings.append((processor.tokenizer.decode([token_id]), time_stamp))

start_time_sec = 30.0
sub_content = ""
index = 1
prev_end_time_sec = start_time_sec  # Initialize with the start time offset
sub_lines = []  # To store tuples of (start_time, end_time, text)

for word, timing in word_timings:
    if not word.strip():
        continue

    word_time_in_seconds = start_time_sec + timing

    # Check if the current word should be appended to the previous line
    if word_time_in_seconds - prev_end_time_sec < 3 and sub_lines:
        # Update the last subtitle line's end time and append the word to its text
        last_start, last_end, last_text = sub_lines[-1]
        updated_text = last_text + " " + word
        sub_lines[-1] = (last_start, word_time_in_seconds, updated_text)
    else:
        # Start a new subtitle line
        sub_lines.append((word_time_in_seconds, word_time_in_seconds, word))

    prev_end_time_sec = word_time_in_seconds

# Convert sub_lines to subtitle format
for start_time_sec, end_time_sec, text in sub_lines:
    from_time = "{:02d}:{:02d}:{:02d},{:03d}".format(
        int(start_time_sec // 3600),
        int(start_time_sec % 3600 // 60),
        int(start_time_sec % 60),
        int((start_time_sec % 1) * 1000),
    )
    to_time = "{:02d}:{:02d}:{:02d},{:03d}".format(
        int(end_time_sec // 3600),
        int(end_time_sec % 3600 // 60),
        int(end_time_sec % 60),
        int((end_time_sec % 1) * 1000),
    )

    sub_content += "{}\n{} --> {}\n{}\n\n".format(index, from_time, to_time, text.strip())
    index += 1

print('\n', sub_content)

# Save to file.sub
with open("file.sub", "w", encoding="utf-8") as sub_file:
    sub_file.write(sub_content)
