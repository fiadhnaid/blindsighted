"""LiveKit Agent for emotion detection from video frames using streaming STT-LLM-TTS pipeline."""

"""
LiveKit Agent for emotion detection from video frames using streaming STT-LLM-TTS pipeline.

PROMPT FOR PER-SECOND FRAME ANALYSIS:
================================================================

You are analyzing a video frame to extract structured information about the person in front of the camera. 
This information will help a blind person understand social cues they cannot see during a conversation.

CRITICAL: Focus ONLY on the main subject - the person closest to the camera, centered in the frame, or the person the camera is clearly focused on.
Ignore other people in the background or periphery. Analyze only the primary subject.

You must respond with ONLY a valid JSON object, no other text. The JSON must follow this exact structure:

{
    "emotion": string,  // One of: "happy", "smiling", "engaged", "interested", "amused", "sad", "frustrated", "confused", "bored", "annoyed", "angry", "neutral", "calm", "focused", "surprised", "embarrassed", "uncertain", "confident", "unknown"
    "person_presence": string,  // One of: "present", "entered", "left", "absent"
    "is_nodding": boolean,  // true if person is nodding their head, false otherwise
    "on_phone": boolean,  // true if person is looking at or using a phone/device, false otherwise
    "confidence": number,  // 0.0 to 1.0, your confidence in this analysis
    "timestamp": string  // ISO 8601 timestamp (you can use current time approximation)
}

Rules:
- "person_presence": Use "present" if a person is clearly visible. Use "entered" only if this is the first frame where a person appears (you may not have context, so default to "present" if unsure). Use "left" only if a person was clearly visible and is now gone. Use "absent" if no person is visible.
- "is_nodding": Look for up-and-down head movements. If you cannot determine from a single frame, use false.
- "on_phone": Detect if the person is holding, looking at, or interacting with a phone or similar device.
- "emotion": Choose the most prominent emotion visible. If multiple emotions are present, choose the dominant one.
- "confidence": Be honest about your confidence. Lower confidence if the person is partially obscured, in poor lighting, or if cues are ambiguous.

If no person is visible or the main subject is unclear, return:
{
    "emotion": "unknown",
    "person_presence": "absent",
    "is_nodding": false,
    "on_phone": false,
    "confidence": 0.0,
    "timestamp": "[current ISO timestamp]"
}

Respond with ONLY the JSON object, no markdown, no code blocks, no explanation.
"""


import asyncio
import json
import logging
import re
from collections import deque
from datetime import datetime

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
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

# Enable debug logging
logging.basicConfig(level=logging.INFO)


class EmotionAssistant(Agent):
    """AI assistant that detects and describes emotions from video frames."""

    # Constants for frame processing pipeline
    FRAME_SAMPLING_INTERVAL = 1.0  # seconds between frame samples
    FRAME_QUEUE_MAX_SIZE = 10  # maximum number of frame descriptions to store

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

        # Frame processing pipeline
        self._frame_descriptions_queue: deque[dict] = deque(maxlen=self.FRAME_QUEUE_MAX_SIZE)
        self._frame_analysis_llm: llm.LLM | None = None
        self._change_detection_llm: llm.LLM | None = None
        self._change_detection_running = False
        self._session: AgentSession | None = None
        self._last_change_detection_result: str | None = None

        # Timestamp-based ordering for frame processing
        self._pending_analyses: list[dict] = []  # list of completed analyses waiting to be inserted

        self._periodic_analysis_task: asyncio.Task | None = None
        self._analysis_interval = 1  # seconds
        logger.info("EmotionAssistant initialized (single person focus)")

    async def on_enter(self) -> None:
        """Called when the agent enters a room. Sets up video stream monitoring."""
        ctx = get_job_context()
        room = ctx.room

        logger.info(f"Emotion agent entered room: {room.name}")

        # Initialize LLMs for frame processing pipeline
        self._frame_analysis_llm = openai.LLM(
            model="google/gemini-2.5-flash",
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            max_completion_tokens=500,
        )

        self._change_detection_llm = openai.LLM(
            model="google/gemini-2.5-flash",
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            max_completion_tokens=300,
        )

        # Start change detection task
        self._start_periodic_change_detection()

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

        # Start periodic frame analysis (every 0.5 seconds)
        self._start_periodic_analysis()
        logger.info(f"Started periodic analysis (interval: {self._analysis_interval}s)")

    async def on_leave(self) -> None:
        """Called when agent leaves the room. Clean up periodic analysis."""
        logger.info("Agent leaving room, stopping periodic analysis")
        self._stop_periodic_analysis()

    async def _periodic_frame_analyzer(self) -> None:
        """
        Continuously analyzes the latest video frame every 1 seconds.

        This runs independently of voice interactions - the frame is analyzed
        on a fixed schedule rather than when the user speaks.
        """
        logger.info("Periodic frame analyzer started")
        analysis_count = 0

        while True:
            try:
                await asyncio.sleep(self._analysis_interval)

                if self._latest_frame is None:
                    logger.debug("No frame available for periodic analysis")
                    continue

                analysis_count += 1
                logger.info(f"Running periodic analysis #{analysis_count}")

                # ============================================================
                # TODO: REPLACE THIS WITH TEAMMATE'S REAL LLM CALL FUNCTION
                # ============================================================
                # The real function should:
                # 1. Take self._latest_frame as input (rtc.VideoFrame)
                # 2. Call your LLM/vision model to analyze the frame
                # 3. Return structured emotion/expression data
                # 4. Handle rate limiting and errors gracefully
                #
                # Example signature:
                # result = await analyze_frame_with_llm(self._latest_frame)
                # ============================================================

                result = await self._dummy_llm_call(self._latest_frame)

                # ============================================================
                # TODO: HANDLE THE RESULT
                # ============================================================
                # Options:
                # 1. Send to iOS via data message:
                #    await ctx.room.local_participant.publish_data(
                #        json.dumps(result).encode(), topic="emotion_analysis"
                #    )
                #
                # 2. Store for later use:
                #    self._last_analysis_result = result
                #
                # 3. Log only (current behavior):
                #    logger.info(f"Analysis result: {result}")
                # ============================================================

                logger.info(f"Periodic analysis result: {result}")

            except asyncio.CancelledError:
                logger.info("Periodic frame analyzer stopped")
                break
            except Exception as e:
                logger.error(f"Error in periodic frame analysis: {e}")
                # Continue running even if one analysis fails
                continue

    async def _dummy_llm_call(self, frame: rtc.VideoFrame) -> dict:
        """
        DUMMY PLACEHOLDER - Replace with real LLM call from teammate.

        This function simulates calling an LLM/vision model to analyze a video frame.

        Args:
            frame: The latest video frame from the glasses camera

        Returns:
            Dict with analysis results (structure TBD by teammate)

        TODO: Replace this entire function with:
        - Real API call to your vision/emotion detection model
        - Proper error handling for rate limits (429 errors)
        - Retry logic with exponential backoff
        - Response validation and parsing
        """
        # Simulate processing time
        await asyncio.sleep(0.01)

        # Return dummy data
        return {
            "status": "dummy_analysis",
            "timestamp": asyncio.get_event_loop().time(),
            "frame_available": frame is not None,
            "message": "Replace _dummy_llm_call() with real LLM integration"
        }

    def _start_periodic_analysis(self) -> None:
        """Start the periodic frame analysis background task."""
        if self._periodic_analysis_task is not None:
            logger.warning("Periodic analysis already running")
            return

        self._periodic_analysis_task = asyncio.create_task(self._periodic_frame_analyzer())
        self._tasks.append(self._periodic_analysis_task)

    def _stop_periodic_analysis(self) -> None:
        """Stop the periodic frame analysis background task."""
        if self._periodic_analysis_task is not None:
            self._periodic_analysis_task.cancel()
            self._periodic_analysis_task = None
            logger.info("Stopped periodic frame analysis")

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

    async def process_image(self, image: rtc.VideoFrame) -> None:
        """Process an image asynchronously: analyze with LLM and add to queue in order.

        This function can be called with any image. It will:
        1. Capture timestamp to preserve order
        2. Process the image with LLM to get JSON description (async)
        3. Add the description to the queue in correct timestamp order when complete
        """
        if not self._frame_analysis_llm:
            logger.warning("Frame analysis LLM not initialized")
            return

        # Capture timestamp when called (preserves order)
        capture_time = datetime.now()

        # Process frame asynchronously (non-blocking)
        asyncio.create_task(self._analyze_frame_and_queue(image, capture_time))

    async def _analyze_frame_and_queue(
        self, frame: rtc.VideoFrame, capture_time: datetime
    ) -> None:
        """Analyze a frame with LLM and add JSON description to queue in correct timestamp order."""
        try:
            chat_ctx = llm.ChatContext()
            messages = [
                llm.ChatMessage(
                    role="user",
                    content=[
                        "Analyze this image and describe the person's emotions and facial expression. Return a JSON object with: apparent_emotion, energy_level, engagement, gaze_direction, facial_cues, gestures, confidence, summary",
                        llm.ImageContent(image=frame),
                    ],
                ),
            ]

            response = await self._frame_analysis_llm.chat(ctx=chat_ctx, chat_messages=messages)
            response_text = " ".join(str(c) for c in response.content)

            # Extract JSON from response
            analysis_dict = self._extract_analysis_dict(response_text)
            if not analysis_dict:
                # Fallback: create minimal dict
                analysis_dict = {
                    "apparent_emotion": "unknown",
                    "energy_level": "unknown",
                    "engagement": "unknown",
                    "gaze_direction": "unknown",
                    "facial_cues": [],
                    "gestures": [],
                    "confidence": 0.0,
                    "summary": "Analysis incomplete",
                }

            # Add timestamp (used for ordering)
            analysis_dict["timestamp"] = capture_time.isoformat()
            analysis_dict["timestamp_obj"] = capture_time  # Keep datetime object for sorting

            # Insert in correct timestamp order
            await self._insert_analysis_in_order(analysis_dict)

        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")

    async def _insert_analysis_in_order(self, analysis_dict: dict) -> None:
        """Insert analysis in correct timestamp order."""
        # Add to pending list
        self._pending_analyses.append(analysis_dict)

        # Sort pending analyses by timestamp
        self._pending_analyses.sort(key=lambda x: x.get("timestamp_obj", datetime.min))

        # Insert all analyses that are in order (timestamp <= last in queue or queue is empty)
        queue_last_timestamp = None
        if self._frame_descriptions_queue:
            last_item = self._frame_descriptions_queue[-1]
            queue_last_timestamp = last_item.get("timestamp_obj")

        # Insert all pending analyses that come after the last queue item
        to_insert = []
        remaining = []
        for analysis in self._pending_analyses:
            analysis_timestamp = analysis.get("timestamp_obj")
            if queue_last_timestamp is None or analysis_timestamp >= queue_last_timestamp:
                to_insert.append(analysis)
            else:
                remaining.append(analysis)

        # Insert in order
        for analysis in to_insert:
            self._frame_descriptions_queue.append(analysis)
            # Remove timestamp_obj (keep only ISO string for JSON serialization)
            analysis.pop("timestamp_obj", None)
            logger.debug(
                f"Frame with timestamp {analysis['timestamp']} inserted. Queue size: {len(self._frame_descriptions_queue)}"
            )

        # Keep remaining analyses that are out of order (waiting for earlier frames)
        self._pending_analyses = remaining

    def _start_periodic_change_detection(self) -> None:
        """Start periodic task to detect changes in frame sequence."""
        if self._change_detection_running:
            return

        self._change_detection_running = True

        async def periodic_change_detection() -> None:
            """Periodically analyze queue for changes."""
            while self._change_detection_running:
                try:
                    await asyncio.sleep(self.FRAME_SAMPLING_INTERVAL)

                    if not self._change_detection_llm or len(self._frame_descriptions_queue) < 2:
                        continue

                    # Get full ordered sequence
                    descriptions = list(self._frame_descriptions_queue)
                    result = await self._detect_changes(descriptions)

                    if result and result != self._last_change_detection_result:
                        self._last_change_detection_result = result
                        await self._send_to_glasses(result)

                except asyncio.CancelledError:
                    self._change_detection_running = False
                    break
                except Exception as e:
                    logger.error(f"Error in change detection: {e}")

        task = asyncio.create_task(periodic_change_detection())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)
        logger.info("Started periodic change detection")

    async def _detect_changes(self, descriptions: list[dict]) -> str | None:
        """Analyze full ordered sequence of frame descriptions for changes."""
        if not self._change_detection_llm or not descriptions:
            return None

        try:
            # Format descriptions with temporal order
            descriptions_text = "\n\n".join(
                f"Frame {i+1} (timestamp: {desc.get('timestamp', 'unknown')}):\n"
                f"Emotion: {desc.get('apparent_emotion', 'unknown')}, "
                f"Energy: {desc.get('energy_level', 'unknown')}, "
                f"Engagement: {desc.get('engagement', 'unknown')}, "
                f"Gaze: {desc.get('gaze_direction', 'unknown')}, "
                f"Summary: {desc.get('summary', 'N/A')}"
                for i, desc in enumerate(descriptions)
            )

            contextual_prompt = """You are analyzing a temporal sequence of frame descriptions. Your task is to:
1. Understand what is currently happening in the scene (based on the latest frames)
2. Identify any significant changes or patterns across the full sequence
3. Consider the progression and order of frames, not just the latest vs previous
4. Focus on contextual clues like: nodding, attention to phone, person approaching/leaving, engagement changes

Return a brief, human-readable string (1-2 sentences) describing:
- What is currently happening in the scene
- Any relevant changes that occurred compared to earlier in the sequence

If no significant changes are detected, return a simple description of the current state only."""

            chat_ctx = llm.ChatContext()
            messages = [
                llm.ChatMessage(
                    role="system",
                    content=[contextual_prompt],
                ),
                llm.ChatMessage(
                    role="user",
                    content=[
                        f"Analyze this sequence of frame descriptions in temporal order:\n\n{descriptions_text}"
                    ],
                ),
            ]

            response = await self._change_detection_llm.chat(ctx=chat_ctx, chat_messages=messages)
            result_text = " ".join(str(c) for c in response.content).strip()

            return result_text if result_text else None

        except Exception as e:
            logger.error(f"Error in change detection: {e}")
            return None

    async def _send_to_glasses(self, text: str) -> None:
        """Send text to glasses via TTS."""
        if not self._session:
            logger.warning("No session available for TTS output")
            return

        try:
            await self._session.generate_reply(instructions=f"Say: {text}")
            logger.info(f"Sent to glasses via TTS: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error sending to glasses: {e}")

    def set_session(self, session: AgentSession) -> None:
        """Set the session reference for TTS output."""
        self._session = session
        logger.info("Session reference set for TTS output")


# Create agent server
server = AgentServer()


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
    agent.set_session(session)  # Provide session reference for TTS output
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

    # Add session TTS event listeners
    if session.tts:
        logger.info("Setting up session TTS event listeners")

        @session.tts.on("error")
        def _on_session_tts_error(error: Exception) -> None:
            logger.warning(f"Session TTS error: {error}")

        @session.tts.on("metrics_collected")
        def _on_session_tts_metrics(metrics) -> None:
            logger.info(f"Session TTS metrics: {metrics}")

    # Start the agent session
    await session.start(room=ctx.room, agent=agent)

    # Generate initial greeting
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


