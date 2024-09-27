import unittest

from unittest.mock import patch, MagicMock
from app.audio_utils import extract_audio


class TestAudioUtils(unittest.TestCase):

    @patch("app.audio_utils.VideoFileClip")
    @patch("app.audio_utils.librosa.load")
    @patch("app.audio_utils.sf.write")
    @patch("app.audio_utils.os.remove")
    def test_extract_audio(
        self, mock_remove, mock_sf_write, mock_librosa_load, mock_video_file_clip
    ):
        # Setup mock objects
        mock_video = MagicMock()
        mock_audio = MagicMock()
        mock_video.audio = mock_audio
        mock_video_file_clip.return_value.__enter__.return_value = mock_video
        mock_librosa_load.return_value = (MagicMock(), 16000)

        # Define test parameters
        videofile_path = "test_video.mp4"
        audio_file_path = "test_audio.wav"
        target_sample_rate = 16000
        output_format = "wav"
        start_time = 10
        end_time = 20
        normalize_audio = True

        # Call the function
        extract_audio(
            videofile_path,
            audio_file_path,
            target_sample_rate,
            output_format,
            start_time,
            end_time,
            normalize_audio,
        )

        # Assertions
        mock_video_file_clip.assert_called_once_with(videofile_path)
        mock_audio.subclip.assert_called_once_with(start_time, end_time)
        mock_audio.write_audiofile.assert_called_once()
        mock_librosa_load.assert_called_once()
        mock_sf_write.assert_called_once_with(
            audio_file_path,
            mock_librosa_load.return_value[0],
            target_sample_rate,
            format=output_format,
        )
        mock_remove.assert_called_once()

    @patch("app.audio_utils.VideoFileClip")
    @patch("app.audio_utils.librosa.load")
    @patch("app.audio_utils.sf.write")
    @patch("app.audio_utils.os.remove")
    def test_extract_audio_no_start_end_time(
        self, mock_remove, mock_sf_write, mock_librosa_load, mock_video_file_clip
    ):
        # Setup mock objects
        mock_video = MagicMock()
        mock_audio = MagicMock()
        mock_video.audio = mock_audio
        mock_video_file_clip.return_value.__enter__.return_value = mock_video
        mock_librosa_load.return_value = (MagicMock(), 16000)

        # Define test parameters
        videofile_path = "test_video.mp4"
        audio_file_path = "test_audio.wav"
        target_sample_rate = 16000
        output_format = "wav"

        # Call the function
        extract_audio(
            videofile_path, audio_file_path, target_sample_rate, output_format
        )

        # Assertions
        mock_video_file_clip.assert_called_once_with(videofile_path)
        mock_audio.subclip.assert_not_called()
        mock_audio.write_audiofile.assert_called_once()
        mock_librosa_load.assert_called_once()
        mock_sf_write.assert_called_once_with(
            audio_file_path,
            mock_librosa_load.return_value[0],
            target_sample_rate,
            format=output_format,
        )
        mock_remove.assert_called_once()

    @patch("app.audio_utils.VideoFileClip")
    @patch("app.audio_utils.librosa.load")
    @patch("app.audio_utils.sf.write")
    @patch("app.audio_utils.os.remove")
    def test_extract_audio_exception(
        self, mock_remove, mock_sf_write, mock_librosa_load, mock_video_file_clip
    ):
        # Setup mock objects
        mock_video_file_clip.side_effect = Exception("Test exception")

        # Define test parameters
        videofile_path = "test_video.mp4"
        audio_file_path = "test_audio.wav"
        target_sample_rate = 16000
        output_format = "wav"

        # Call the function and assert exception
        with self.assertRaises(Exception):
            extract_audio(
                videofile_path, audio_file_path, target_sample_rate, output_format
            )

        # Assertions
        mock_video_file_clip.assert_called_once_with(videofile_path)
        mock_audio = mock_video_file_clip.return_value.__enter__.return_value.audio
        mock_audio.subclip.assert_not_called()
        mock_audio.write_audiofile.assert_not_called()
        mock_librosa_load.assert_not_called()
        mock_sf_write.assert_not_called()
        mock_remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
