
!pip install mecab-python3
!pip install unidic-lite
!python -m unidic download

https://huggingface.co/vumichien/wav2vec2-large-xlsr-japanese

https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html


# So the plan

1. Cut the audio into 10s chunks
2. Generate sub for each chunk
3. Merge the sub into a single file


