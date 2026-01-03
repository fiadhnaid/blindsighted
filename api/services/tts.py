from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from config import settings


class TextToSpeechService:
    """Service for converting text to speech using ElevenLabs"""

    def __init__(self) -> None:
        if not settings.elevenlabs_api_key:
            raise ValueError("ElevenLabs API key not configured")

        self.client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        self.voice_id = settings.elevenlabs_voice_id

    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech audio using ElevenLabs

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as bytes (MP3 format)
        """
        try:
            # Generate audio using ElevenLabs streaming API
            audio_generator = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    use_speaker_boost=True,
                ),
            )

            # Collect all audio chunks
            audio_chunks = list(audio_generator)
            return b"".join(audio_chunks)

        except Exception as e:
            print(f"Error generating speech: {e}")
            raise


tts_service = TextToSpeechService()
