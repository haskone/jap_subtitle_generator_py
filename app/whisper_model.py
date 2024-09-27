import torch
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor, pipeline


class WhisperTranscriber:
    def __init__(self, model_id="openai/whisper-small", use_cuda=True):
        self.device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
        self.torch_dtype = (
            torch.float16 if use_cuda and torch.cuda.is_available() else torch.float32
        )
        self.model_id = model_id

        # Lazy loading of model and processor
        self._model = None
        self._processor = None
        self._pipe = None

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            ).to(self.device)
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._processor = WhisperProcessor.from_pretrained(
                self.model_id, language="ja", task="transcribe"
            )
        return self._processor

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=self.torch_dtype,
                device=self.device,
                generate_kwargs={"language": "ja"},
            )
        return self._pipe

    def transcribe(self, audio_file):
        return self.pipe(audio_file)


# TODO: May try "openai/whisper-large-v3" for better results
transcriber = WhisperTranscriber()


def transcribe(audio_file):
    return transcriber.transcribe(audio_file)


def set_model(model_id="openai/whisper-small", use_cuda=True):
    global transcriber
    transcriber = WhisperTranscriber(model_id, use_cuda)
