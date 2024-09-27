import unittest

from app.translation_model import (
    TranslationModel,
    set_translation_model,
    translate_batch,
)


class TestTranslationModel(unittest.TestCase):
    def setUp(self):
        self.translator = TranslationModel()

    def test_tokenizer_initialization(self):
        tokenizer = self.translator.tokenizer
        self.assertIsNotNone(tokenizer)
        self.assertEqual(tokenizer.name_or_path, self.translator.model_name)

    def test_model_initialization(self):
        model = self.translator.model
        self.assertIsNotNone(model)
        self.assertEqual(model.name_or_path, self.translator.model_name)

    def test_pipeline_initialization(self):
        pipeline = self.translator.pipeline
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.task, "translation_ja_to_en")

    def test_translate_one(self):
        result = self.translator.translate_one("こんにちは")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("translation_text", result[0])

    def test_translate_batch(self):
        texts = ["こんにちは", "おはようございます"]
        results = self.translator.translate_batch(texts)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(texts))
        for result in results:
            self.assertIsInstance(result, str)

    def test_set_translation_model(self):
        set_translation_model("Helsinki-NLP/opus-mt-en-de", use_cuda=False)
        self.assertEqual(self.translator.model_name, "Helsinki-NLP/opus-mt-en-de")
        self.assertEqual(self.translator.device, "cpu")

    def test_translate_batch_function(self):
        texts = ["こんにちは", "おはようございます"]
        results = translate_batch(texts)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(texts))
        for result in results:
            self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
