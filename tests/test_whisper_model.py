import unittest

from unittest.mock import patch, MagicMock
from app.whisper_model import WhisperTranscriber, transcribe, set_model


class TestWhisperTranscriber(unittest.TestCase):

    @patch("app.whisper_model.AutoModelForSpeechSeq2Seq.from_pretrained")
    @patch("app.whisper_model.WhisperProcessor.from_pretrained")
    @patch("app.whisper_model.pipeline")
    def setUp(
        self, mock_pipeline, mock_processor_from_pretrained, mock_model_from_pretrained
    ):
        self.mock_model = MagicMock()
        self.mock_processor = MagicMock()
        self.mock_pipeline = MagicMock()

        mock_model_from_pretrained.return_value = self.mock_model
        mock_processor_from_pretrained.return_value = self.mock_processor
        mock_pipeline.return_value = self.mock_pipeline

        self.transcriber = WhisperTranscriber()

    def test_model_loading(self):
        model = self.transcriber.model
        self.assertIsNotNone(model)
        self.assertEqual(model, self.mock_model)

    def test_processor_loading(self):
        processor = self.transcriber.processor
        self.assertIsNotNone(processor)
        self.assertEqual(processor, self.mock_processor)

    def test_pipeline_loading(self):
        pipe = self.transcriber.pipe
        self.assertIsNotNone(pipe)
        self.assertEqual(pipe, self.mock_pipeline)

    def test_transcribe(self):
        audio_file = "dummy_audio_file.wav"
        self.transcriber.pipe.return_value = "transcription_result"
        result = self.transcriber.transcribe(audio_file)
        self.assertEqual(result, "transcription_result")
        self.transcriber.pipe.assert_called_once_with(audio_file)

    @patch("app.whisper_model.WhisperTranscriber.transcribe")
    def test_transcribe_function(self, mock_transcribe):
        audio_file = "dummy_audio_file.wav"
        mock_transcribe.return_value = "transcription_result"
        result = transcribe(audio_file)
        self.assertEqual(result, "transcription_result")
        mock_transcribe.assert_called_once_with(audio_file)

    @patch("app.whisper_model.WhisperTranscriber.__init__", return_value=None)
    def test_set_model(self, mock_init):
        set_model("openai/whisper-large-v3", use_cuda=False)
        mock_init.assert_called_once_with("openai/whisper-large-v3", use_cuda=False)


if __name__ == "__main__":
    unittest.main()
