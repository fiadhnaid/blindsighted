"""LiveKit Agent for emotion detection from video frames using streaming STT-LLM-TTS pipeline."""

import asyncio
import json
import re

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    get_job_context,
    llm,
)
from livekit.agents.voice.events import ConversationItemAddedEvent, SpeechCreatedEvent
from livekit.plugins import deepgram, elevenlabs, openai, silero
from loguru import logger

from config import settings


class EmotionAssistant(Agent):
    """AI assistant that detects and describes emotions from video frames."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a facial expression and behavior analysis AI assistant. Your primary task is to analyze images
            (photos or video frames) and provide comprehensive insights about THE PERSON DIRECTLY IN FRONT OF THE CAMERA.

            CRITICAL: Focus ONLY on the main subject - the person closest to the camera, centered in the frame, or the person the camera is clearly focused on.
            Ignore other people in the background or periphery. Analyze only the primary subject.

            IMPORTANT: You must respond with BOTH a natural language description AND a JSON object.

            Format your response as:
            [DESCRIPTION]
            Provide a brief, natural 2-3 sentence description of what you observe about the person's facial expression, energy, engagement, and overall demeanor. Be conversational and human-friendly. This will be read aloud to the user.

            [JSON]
            {
                "apparent_emotion": string, // e.g. "relaxed", "tense", "bored", "amused", "frustrated", "uncertain", "happy", "sad", "angry", "surprised", "neutral"
                "energy_level": string, // one of ["low","medium","high"]
                "engagement": string, // one of ["disengaged","neutral","attentive","very_attentive"]
                "gaze_direction": string, // one of ["camera","left","right","up","down","unfocused","unknown"]
                "facial_cues": [string], // short phrases, e.g. ["slight smile","raised eyebrows","tight jaw","furrowed brow","wide eyes"]
                "gestures": [
                    {
                        "type": string, // e.g. "nod","head_shake","lean_in","eyebrow_raise","smile_change","frown","eye_roll"
                        "description": string, // short natural-language description
                        "confidence": number // 0.0 to 1.0
                    }
                ],
                "confidence": number, // 0.0 to 1.0 overall confidence in your interpretation
                "summary": string // max 2 sentences, informal human-friendly description
            }

            If no person is visible or the main subject is unclear, return:
            [DESCRIPTION]
            I cannot see a clear person or face in the image right now.

            [JSON]
            {
                "apparent_emotion": "unknown",
                "energy_level": "unknown",
                "engagement": "unknown",
                "gaze_direction": "unknown",
                "facial_cues": [],
                "gestures": [],
                "confidence": 0.0,
                "summary": "No clear person or face visible in the image"
            }

            Analyze the MAIN SUBJECT carefully:
            - Facial expressions (smile, frown, raised eyebrows, etc.)
            - Eye contact and gaze direction (especially if looking at camera)
            - Head position and orientation
            - Body language visible in frame
            - Overall energy and engagement level
            - Subtle micro-expressions if visible

            The [DESCRIPTION] section will be read aloud, so make it natural and conversational.
            The [JSON] section is for programmatic access.
            """
        )
        self._latest_frame: rtc.VideoFrame | None = None
        self._video_stream: rtc.VideoStream | None = None
        self._tasks: list[asyncio.Task] = []
        self._last_analysis_dict: dict | None = None
        logger.info("EmotionAssistant initialized (single person focus)")

    async def on_enter(self) -> None:
        """Called when the agent enters a room. Sets up video stream monitoring."""
        ctx = get_job_context()
        room = ctx.room

        logger.info(f"Emotion agent entered room: {room.name}")

        # Find the first video track from remote participant (if any)
        if room.remote_participants:
            remote_participant = list(room.remote_participants.values())[0]
            video_tracks = [
                publication.track
                for publication in list(remote_participant.track_publications.values())
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO
            ]
            if video_tracks:
                self._create_video_stream(video_tracks[0])
                logger.info(
                    f"Subscribed to existing video track from {remote_participant.identity}"
                )

        # Watch for new video tracks not yet published
        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            """Handle new track subscription."""
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                logger.info(f"New video track subscribed from {participant.identity}")
                self._create_video_stream(track)

    async def on_user_turn_completed(
        self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Add the latest video frame to the user's message for facial expression analysis."""
        if self._latest_frame:
            new_message.content.append(llm.ImageContent(image=self._latest_frame))
        else:
            logger.warning("No video frame available")
            new_message.content.append(
                "[SYSTEM: No video frame available. Return [DESCRIPTION] I cannot see your camera feed right now. [JSON] {\"apparent_emotion\": \"unknown\", \"energy_level\": \"unknown\", \"engagement\": \"unknown\", \"gaze_direction\": \"unknown\", \"facial_cues\": [], \"gestures\": [], \"confidence\": 0.0, \"summary\": \"No video feed available\"}]"
            )

    async def on_agent_response(
        self, chat_ctx: llm.ChatContext, response: llm.ChatMessage
    ) -> None:
        """Modify response content before TTS - extract only the natural language description."""
        if not response.content:
            return

        response_text = " ".join(str(c) for c in response.content)

        # Extract natural language description for TTS
        description_match = re.search(
            r'\[DESCRIPTION\]\s*(.*?)(?=\[JSON\]|$)', response_text, re.IGNORECASE | re.DOTALL
        )
        if description_match:
            description = description_match.group(1).strip()
            response.content = [llm.TextContent(text=description)]
        else:
            json_start = re.search(r'\[JSON\]', response_text, re.IGNORECASE)
            if json_start:
                description = response_text[:json_start.start()].strip()
                description = re.sub(r'\[DESCRIPTION\]\s*', '', description, flags=re.IGNORECASE).strip()
                if description:
                    response.content = [llm.TextContent(text=description)]

        # Extract and store JSON for programmatic access
        analysis_dict = self._extract_analysis_dict(response_text)
        if analysis_dict:
            self._last_analysis_dict = analysis_dict


    def _extract_analysis_dict(self, text: str) -> dict | None:
        """Extract JSON from text response. Returns a single dict or None."""
        # First, try to extract JSON from [JSON] section if present
        json_section_match = re.search(
            r'\[JSON\]\s*(\{[\s\S]*?\})', text, re.IGNORECASE | re.DOTALL
        )
        if json_section_match:
            try:
                parsed = json.loads(json_section_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
                elif isinstance(parsed, list) and len(parsed) > 0:
                    logger.warning("Received array instead of single object, using first element")
                    return parsed[0] if isinstance(parsed[0], dict) else None
            except json.JSONDecodeError:
                pass

        # Try to find JSON object with balanced braces
        brace_count = 0
        start_idx = text.find('{')
        if start_idx != -1:
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON object
                        try:
                            parsed = json.loads(text[start_idx:i+1])
                            if isinstance(parsed, dict):
                                return parsed
                            elif isinstance(parsed, list) and len(parsed) > 0:
                                logger.warning("Received array instead of single object, using first element")
                                return parsed[0] if isinstance(parsed[0], dict) else None
                        except json.JSONDecodeError:
                            pass
                        break

        # Fallback: parse entire response as JSON
        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                return parsed
            elif isinstance(parsed, list) and len(parsed) > 0:
                logger.warning("Received array instead of single object, using first element")
                return parsed[0] if isinstance(parsed[0], dict) else None
        except json.JSONDecodeError:
            return None

        return None

    @property
    def last_analysis_dict_str(self) -> str | None:
        """Get the last facial expression analysis as a formatted JSON string."""
        if self._last_analysis_dict:
            return json.dumps(self._last_analysis_dict, indent=2)
        return None

    def get_facial_analysis(self) -> dict | None:
        """Get the latest facial expression analysis results for the main subject.
        Returns:
            {
                "apparent_emotion": "...",
                "energy_level": "...",
                "engagement": "...",
                "gaze_direction": "...",
                "facial_cues": [...],
                "gestures": [...],
                "confidence": 0.0-1.0,
                "summary": "..."
            }
            Returns None if no analysis available.
        """
        return self._last_analysis_dict

    def _create_video_stream(self, track: rtc.Track) -> None:
        """Create a video stream to buffer the latest frame."""
        if self._video_stream is not None:
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            self._tasks.clear()

        self._video_stream = rtc.VideoStream(track)

        async def read_stream() -> None:
            if not self._video_stream:
                return
            async for event in self._video_stream:
                self._latest_frame = event.frame

        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)


async def should_accept_job(job_request: JobRequest) -> None:
    """Filter function to accept only jobs matching this agent's name.

    The agent name is configured via settings.livekit_agent_name
    and should match the agent_id stored in the room metadata by the API.
    """
    agent_name = settings.livekit_agent_name
    room_metadata = job_request.room.metadata

    # If no agent name is configured in the room metadata, accept all jobs (backward compatibility)
    if not room_metadata:
        logger.warning(
            f"Room {job_request.room.name} has no metadata - accepting job for backward compatibility"
        )
        await job_request.accept()
        return

    if room_metadata == agent_name:
        logger.info(f"Accepting job for room {job_request.room.name} (agent: {agent_name})")
        await job_request.accept()
    else:
        logger.info(f"Skipping job for room {job_request.room.name} (expected: {agent_name}, got: {room_metadata})")


async def entrypoint(ctx: JobContext) -> None:
    """Entry point for the emotion detection agent.

    Uses streaming STT-LLM-TTS pipeline with vision capabilities for emotion analysis.
    """
    logger.info(f"Starting emotion detection agent for room: {ctx.room.name}")

    await ctx.connect()

    # Use ElevenLabs TTS for natural, expressive voice
    tts_instance = elevenlabs.TTS(
        api_key=settings.elevenlabs_api_key,
        voice_id=settings.elevenlabs_voice_id,
        model="eleven_turbo_v2_5",
    )

    # Configure the agent session with STT-LLM-TTS pipeline
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            api_key=settings.deepgram_api_key,
        ),
        llm=openai.LLM(
            model="google/gemini-2.5-flash",
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            max_completion_tokens=500,
        ),
        tts=tts_instance,
        vad=silero.VAD.load(),
        min_interruption_duration=1.0,
        allow_interruptions=True,
        use_tts_aligned_transcript=True,
    )

    agent = EmotionAssistant()
    await session.start(room=ctx.room, agent=agent)
    @session.on("user_input_transcribed")
    def _on_user_input(text: str) -> None:
        logger.info(f"User said: {text}")

    @session.on("speech_created")
    def _on_speech_created(event: SpeechCreatedEvent) -> None:
        handle = event.speech_handle
        logger.info(f"Speech from {event.source} with handle #{handle.id}")

    @session.on("conversation_item_added")
    def _on_conversation_item(event: ConversationItemAddedEvent) -> None:
        item = event.item
        if isinstance(item, llm.ChatMessage):
            content = item.content[0] if item.content else ""
            logger.info(f"Conversation item: role={item.role}, content: '{content}'")

    if session.tts:
        @session.tts.on("error")
        def _on_session_tts_error(error: Exception) -> None:
            logger.warning(f"TTS error: {error}")

    await session.generate_reply(instructions="Say 'emotion detection ready'.")

    logger.info("Emotion detection agent session started successfully")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=should_accept_job,
            ws_url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
        )
    )

