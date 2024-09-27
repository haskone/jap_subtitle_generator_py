from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Pipeline
import torch
from functools import lru_cache


class TranslationModel:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-ja-en", use_cuda=True):
        self.model_name = model_name
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._model = None
        self._pipeline = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(
                self.device
            )
        return self._model

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = pipeline(
                "translation_ja_to_en",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        return self._pipeline

    @lru_cache(maxsize=1000)
    def translate_one(self, japanese_text) -> Pipeline:
        return self.pipeline(japanese_text, max_length=512)

    def translate_batch(self, japanese_texts, batch_size=32) -> list[str]:
        results = []
        for i in range(0, len(japanese_texts), batch_size):
            batch = japanese_texts[i : i + batch_size]
            translated = self.translate_one(batch)
            results.extend([t["translation_text"] for t in translated])
        return results


translator = TranslationModel()


def translate_batch(japanese_texts, batch_size=32) -> list[str]:
    return translator.translate_batch(japanese_texts, batch_size)


def set_translation_model(model_name="Helsinki-NLP/opus-mt-ja-en", use_cuda=True):
    global translator
    translator = TranslationModel(model_name, use_cuda)
