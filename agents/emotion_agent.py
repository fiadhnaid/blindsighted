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
    FRAME_SAMPLING_INTERVAL = 2.0  # seconds between frame samples
    FRAME_QUEUE_MAX_SIZE = 10  # maximum number of frame descriptions to store

    def __init__(self) -> None:
        # Main agent instructions (for voice conversation - returns description + JSON)
        # Uses the same schema as the periodic frame analysis for consistency
        conversation_instructions = """You are analyzing a video frame to help a blind person understand social cues during a conversation.

CRITICAL: Focus ONLY on the main subject - the person closest to the camera, centered in the frame, or the person the camera is clearly focused on.
Ignore other people in the background or periphery. Analyze only the primary subject.

IMPORTANT: You must respond with BOTH a natural language description AND a JSON object.

Format your response as:
[DESCRIPTION]
Provide a brief, natural 2-3 sentence description of what you observe about the person's emotion, whether they're nodding, if they're on their phone, and any other relevant social cues. Be conversational and human-friendly. This will be read aloud to the user.

[JSON]
{
    "emotion": string,  // One of: "happy", "smiling", "engaged", "interested", "amused", "sad", "frustrated", "confused", "bored", "annoyed", "angry", "neutral", "calm", "focused", "surprised", "embarrassed", "uncertain", "confident", "unknown"
    "person_presence": string,  // One of: "present", "entered", "left", "absent"
    "is_nodding": boolean,  // true if person is nodding their head, false otherwise
    "on_phone": boolean,  // true if person is looking at or using a phone/device, false otherwise
    "confidence": number,  // 0.0 to 1.0, your confidence in this analysis
    "timestamp": string  // ISO 8601 timestamp
}

If no person is visible or the main subject is unclear, return:
[DESCRIPTION]
I cannot see a clear person or face in the image right now.

[JSON]
{
    "emotion": "unknown",
    "person_presence": "absent",
    "is_nodding": false,
    "on_phone": false,
    "confidence": 0.0,
    "timestamp": "[current ISO timestamp]"
}

The [DESCRIPTION] section will be read aloud, so make it natural and conversational.
The [JSON] section is for programmatic access.
"""
        super().__init__(instructions=conversation_instructions)
        self._latest_frame: rtc.VideoFrame | None = None
        self._video_stream: rtc.VideoStream | None = None
        self._tasks: list[asyncio.Task] = []

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
        self._analysis_interval = 2  # seconds
        logger.info("EmotionAssistant initialized (single person focus)")

    async def on_enter(self) -> None:
        """Called when the agent enters a room. Sets up video stream monitoring."""
        ctx = get_job_context()
        room = ctx.room

        logger.info("=" * 70)
        logger.info(f"🚀 Emotion agent initializing for room: {room.name}")
        logger.info("=" * 70)

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
        logger.info("=" * 70)
        logger.info("✓ Emotion agent initialization complete - ready to analyze frames")
        logger.info("=" * 70)

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
                logger.info(f"==================== Periodic Analysis #{analysis_count} ====================")

                # Call real LLM analysis - this will analyze the frame and add result to queue
                await self.process_image(self._latest_frame)
                logger.info(f"Triggered frame analysis (queue size: {len(self._frame_descriptions_queue)})")
                logger.info("=" * 70)

            except asyncio.CancelledError:
                logger.info("Periodic frame analyzer stopped")
                break
            except Exception as e:
                logger.error(f"Error in periodic frame analysis: {e}")
                # Continue running even if one analysis fails
                continue


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
                "[SYSTEM: No video frame available. Return [DESCRIPTION] I cannot see your camera feed right now. [JSON] {\"emotion\": \"unknown\", \"person_presence\": \"absent\", \"is_nodding\": false, \"on_phone\": false, \"confidence\": 0.0, \"timestamp\": \"[current]\"}]"
            )

    async def on_agent_response(
        self, chat_ctx: llm.ChatContext, response: llm.ChatMessage
    ) -> None:
        """Modify response content before TTS - extract only the natural language description."""
        if not response.content:
            return

        response_text = " ".join(str(c) for c in response.content)

        # Extract natural language description for TTS (strip out JSON)
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
            # Frame analysis instructions - using prompt from top of file
            frame_analysis_prompt = """You are analyzing a video frame to extract structured information about the person in front of the camera.
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

Respond with ONLY the JSON object, no markdown, no code blocks, no explanation."""

            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(
                role="system",
                content=[frame_analysis_prompt],
            )
            chat_ctx.add_message(
                role="user",
                content=[
                    "Analyze this image and return the JSON object describing the person's emotions and facial expression.",
                    llm.ImageContent(image=frame),
                ],
            )

            logger.info("Calling frame analysis LLM...")
            stream = self._frame_analysis_llm.chat(chat_ctx=chat_ctx)

            # Collect full response - to_str_iterable() converts stream to text chunks
            response_text = ""
            async for text_chunk in stream.to_str_iterable():
                response_text += text_chunk

            logger.info(f"Frame analysis LLM response received (length: {len(response_text)} chars)")
            logger.debug(f"Full response: {response_text[:200]}...")

            # Extract JSON from response
            analysis_dict = self._extract_analysis_dict(response_text)
            if not analysis_dict:
                logger.warning("Failed to extract JSON from LLM response, using fallback")
                # Fallback: create minimal dict matching the schema
                analysis_dict = {
                    "emotion": "unknown",
                    "person_presence": "absent",
                    "is_nodding": False,
                    "on_phone": False,
                    "confidence": 0.0,
                    "timestamp": capture_time.isoformat(),
                }
            else:
                logger.info(
                    f"Frame analysis complete - Emotion: {analysis_dict.get('emotion')}, "
                    f"Person: {analysis_dict.get('person_presence')}, "
                    f"Nodding: {analysis_dict.get('is_nodding')}, "
                    f"On phone: {analysis_dict.get('on_phone')}, "
                    f"Confidence: {analysis_dict.get('confidence')}"
                )

            # Add timestamp (used for ordering)
            analysis_dict["timestamp"] = capture_time.isoformat()
            analysis_dict["timestamp_obj"] = capture_time  # Keep datetime object for sorting

            # Insert in correct timestamp order
            await self._insert_analysis_in_order(analysis_dict)
            logger.info(f"Frame description added to queue (queue size now: {len(self._frame_descriptions_queue)})")

        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")

    async def _insert_analysis_in_order(self, analysis_dict: dict) -> None:
        """Insert analysis in correct timestamp order by merging and sorting queue + pending."""
        # Add to pending list
        self._pending_analyses.append(analysis_dict)

        # Merge queue and pending, then sort by timestamp
        all_analyses = list(self._frame_descriptions_queue) + self._pending_analyses

        # Sort by timestamp_obj (items in queue may not have it, so parse from ISO string)
        def get_timestamp(analysis: dict) -> datetime:
            timestamp_obj = analysis.get("timestamp_obj")
            if timestamp_obj:
                return timestamp_obj
            # Parse from ISO string if timestamp_obj was removed
            if "timestamp" in analysis:
                try:
                    return datetime.fromisoformat(analysis["timestamp"])
                except (ValueError, TypeError):
                    pass
            return datetime.min

        all_analyses.sort(key=get_timestamp)

        # Rebuild queue with sorted items (respecting maxlen)
        self._frame_descriptions_queue.clear()
        for analysis in all_analyses[-self.FRAME_QUEUE_MAX_SIZE :]:
            # Keep timestamp_obj for future sorting (don't remove it)
            self._frame_descriptions_queue.append(analysis)
            logger.debug(
                f"Frame with timestamp {analysis['timestamp']} in queue. Queue size: {len(self._frame_descriptions_queue)}"
            )

        # Clear pending - all items are now in queue (or dropped if over maxlen)
        self._pending_analyses.clear()

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

                    if not self._change_detection_llm:
                        logger.debug("Change detection LLM not initialized")
                        continue

                    if len(self._frame_descriptions_queue) < 2:
                        logger.debug(f"Not enough frames in queue for change detection (need 2, have {len(self._frame_descriptions_queue)})")
                        continue

                    # Get only the last 5 descriptions (most recent context)
                    descriptions = list(self._frame_descriptions_queue)[-5:]
                    logger.info(f"---------- Change Detection (analyzing {len(descriptions)} frames) ----------")
                    result = await self._detect_changes(descriptions)

                    # result is now a tuple: (should_speak: bool, description: str)
                    if result:
                        should_speak, description = result
                        logger.info(f"Change detection result - Should speak: {should_speak}, Description: {description[:100]}...")

                        # Only send to glasses if LLM says we should speak
                        if should_speak and description != self._last_change_detection_result:
                            self._last_change_detection_result = description
                            logger.info(f"✓ SPEAKING TO USER: {description}")
                            await self._send_to_glasses(description)
                        elif should_speak and description == self._last_change_detection_result:
                            logger.info("Skipping - same description as last time (no change)")
                        elif not should_speak:
                            logger.info(f"Skipping - not significant enough to speak: {description[:100]}...")
                    else:
                        logger.warning("Change detection returned None (LLM error or no result)")

                except asyncio.CancelledError:
                    self._change_detection_running = False
                    break
                except Exception as e:
                    logger.error(f"Error in change detection: {e}")

        task = asyncio.create_task(periodic_change_detection())
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        self._tasks.append(task)
        logger.info("Started periodic change detection")

    async def _detect_changes(self, descriptions: list[dict]) -> tuple[bool, str] | None:
        """Analyze full ordered sequence of frame descriptions for changes.

        Returns:
            tuple[bool, str] where:
                - bool: True if this observation should be spoken to the user
                - str: The human-readable description of what's happening
            Returns None if analysis fails or no descriptions available
        """
        if not self._change_detection_llm or not descriptions:
            return None

        try:
            # Format descriptions with temporal order using new schema
            descriptions_text = "\n\n".join(
                f"Frame {i+1} (timestamp: {desc.get('timestamp', 'unknown')}):\n"
                f"Emotion: {desc.get('emotion', 'unknown')}, "
                f"Person presence: {desc.get('person_presence', 'unknown')}, "
                f"Nodding: {desc.get('is_nodding', False)}, "
                f"On phone: {desc.get('on_phone', False)}, "
                f"Confidence: {desc.get('confidence', 0.0)}"
                for i, desc in enumerate(descriptions)
            )

            contextual_prompt = """You are analyzing a sequence of JSON analysis results from video frame processing.
Your task is to compare the most recent JSON (last one) with the previous JSONs to detect if there are meaningful changes that should be spoken aloud to a visually impaired user wearing smart glasses.

INPUT:
You will receive 5 frame descriptions in temporal order. The last frame is the most recent analysis.
Each frame description contains:
- emotion: The person's emotional state
- person_presence: Whether a person is present, entered, left, or absent
- is_nodding: Whether the person is nodding their head (boolean)
- on_phone: Whether the person is looking at/using a phone (boolean)
- confidence: Confidence score (0.0 to 1.0)
- timestamp: When the frame was captured

YOUR TASK:
1. Compare all frames with the previous frames
2. Determine if there is a SIGNIFICANT change that warrants speaking to the user
3. If there is a significant change, write the EXACT words that should be spoken through the glasses
4. Ignore minor variations in confidence scores or timestamp differences
5. It is very important to let the user know what is happening in the scene.
6. The user is blind and relies on the glasses to know what is happening.
7. The user is not able to see the screen of the glasses, so they rely on the voice to know what is happening.
8. The user is not able to see the screen of the glasses, so they rely on the voice to know what is happening.
9. The user is not able to see the screen of the glasses, so they rely on the voice to know what is happening.
13. Let the user know what is happening in the scene.
14. Be concise and to the point in your output, we prefer short and concise descriptions.
15. Be friendly and engaging in your output.
OUTPUT FORMAT:
You must respond with EXACTLY this format (no other text):

SHOULD_SPEAK: true
DESCRIPTION: [the exact words to speak - written as if talking directly to the user]

OR

SHOULD_SPEAK: false
DESCRIPTION: [brief note for logging - this will NOT be spoken]

CRITICAL: When SHOULD_SPEAK is true, the DESCRIPTION is the EXACT text that will be converted to speech and played through the user's Ray-Ban glasses. Write it as if you are speaking directly to the blind user in a natural, conversational way. 


CRITICAL: KEEP DESCRIPTIONS SHORT AND CONCISE. Maximum 1 sentence. Be brief and direct. The user needs quick, actionable information, not lengthy explanations.

RULES FOR SHOULD_SPEAK:

Set SHOULD_SPEAK to true ONLY if there is a SIGNIFICANT change:
- The person's emotion clearly changed (e.g., "neutral" -> "happy", "engaged" -> "bored", "calm" -> "frustrated")
- Person presence status changed (e.g., "absent" -> "present", "present" -> "left")
- Behavioral state changed (e.g., is_nodding: false -> true, on_phone: false -> true)
- A behavior stopped (e.g., was nodding and now stopped, was on phone and now put it away)
- The person entered or left the frame

Set SHOULD_SPEAK to false if:
- The JSONs are essentially the same across all 5 frames
- Only confidence scores changed slightly
- Only timestamps differ
- Minor variations in emotion that might be due to detection uncertainty (e.g., "happy" vs "smiling" when both indicate positive emotion)
- The person was already absent and remains absent
- The state in frame 5 matches the state in most of frames 1-4 (no new change)
- The change is too subtle or uncertain to be meaningful

DESCRIPTION GUIDELINES:

When SHOULD_SPEAK is true (THIS WILL BE SPOKEN TO THE USER):
- MUST be SHORT and CONCISE - maximum 1 sentence, ideally under 10 words
- Write as if speaking directly to the user in a natural, conversational tone
- Be precise and specific about what changed
- Use simple, clear language suitable for audio playback
- State what changed clearly and briefly (e.g., "The person started nodding" not "Nodding state changed from false to true")
- Examples of good SHORT speech output:
  * "The person started nodding."
  * "They're looking at their phone now."
  * "The person is smiling now."
  * "They stopped nodding."
  * "They put away their phone."
  * "Someone new entered."
  * "The person left."

- AVOID lengthy descriptions like:
  * "The person's emotional state has transitioned from a neutral expression to one that appears happy, and they are now displaying a smile." (TOO LONG)
  * Instead say: "The person is smiling now." (SHORT)

When SHOULD_SPEAK is false (THIS IS FOR LOGGING ONLY, NOT SPOKEN):
- Briefly note the current stable state
- Examples: "No changes detected." or "Person remains neutral."

COMPARISON LOGIC:
- Compare the last frames against the pattern/consensus of the previous frames
- If the reference frames show a stable state (e.g., all show "neutral" emotion, not nodding) and the last frame differs (e.g., "happy", nodding), that's a change → SHOULD_SPEAK: true
- If the reference frames already showed variation and last frame matches one of those states, that's NOT a new change → SHOULD_SPEAK: false
- If the final frame matches the majority state of previous frames, that's NOT a change (it's continuation) → SHOULD_SPEAK: false

Remember: When SHOULD_SPEAK is true, write the DESCRIPTION as if you are the voice speaking directly to the blind user through their glasses. Be natural, clear, helpful, and MOST IMPORTANTLY - KEEP IT SHORT AND CONCISE."""

            logger.debug(f"Change detection input:\n{descriptions_text}")

            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(
                role="system",
                content=[contextual_prompt],
            )
            chat_ctx.add_message(
                role="user",
                content=[
                    f"Analyze this sequence of frame descriptions in temporal order:\n\n{descriptions_text}"
                ],
            )

            logger.info("Calling change detection LLM...")
            stream = self._change_detection_llm.chat(chat_ctx=chat_ctx)

            # Collect full response
            result_text = ""
            async for text_chunk in stream.to_str_iterable():
                result_text += text_chunk

            result_text = result_text.strip()
            logger.info(f"Change detection LLM raw response: {result_text}")

            # Parse the response to extract SHOULD_SPEAK and DESCRIPTION
            should_speak = False
            description = ""

            # Look for SHOULD_SPEAK: true/false
            should_speak_match = re.search(r'SHOULD_SPEAK:\s*(true|false)', result_text, re.IGNORECASE)
            if should_speak_match:
                should_speak = should_speak_match.group(1).lower() == "true"
                logger.debug(f"Parsed SHOULD_SPEAK: {should_speak}")
            else:
                logger.warning("Could not find SHOULD_SPEAK in response, defaulting to false")

            # Look for DESCRIPTION: ...
            description_match = re.search(r'DESCRIPTION:\s*(.+)', result_text, re.IGNORECASE | re.DOTALL)
            if description_match:
                description = description_match.group(1).strip()
                # Remove any leading "description:" or "description -" that LLM might redundantly add
                description = re.sub(r'^description[\s:;,\-]*', '', description, flags=re.IGNORECASE).strip()
                logger.debug(f"Parsed DESCRIPTION (after cleanup): {description[:100]}...")
            else:
                # Fallback: use entire text if no DESCRIPTION marker found
                logger.warning("Could not find DESCRIPTION marker, using entire response")
                description = result_text
                # Clean up fallback too
                description = re.sub(r'^description[\s:;,\-]*', '', description, flags=re.IGNORECASE).strip()

            if not description:
                logger.warning("No description found in change detection response")
                return None

            return (should_speak, description)

        except Exception as e:
            logger.error(f"Error in change detection: {e}")
            return None

    async def _send_to_glasses(self, text: str) -> None:
        """Send text to glasses via TTS."""
        if not self._session:
            logger.warning("No session available for TTS output")
            return

        try:
            logger.info(f"🔊 Generating TTS for: '{text}'")
            await self._session.generate_reply(instructions=f"Say: {text}")
            logger.info(f"✓ TTS sent successfully")
        except Exception as e:
            logger.error(f"❌ Error sending to glasses: {e}")

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
        allow_interruptions=False,
        use_tts_aligned_transcript=True,
    )

    # Set up event listeners before starting session
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
    agent = EmotionAssistant()
    await session.start(room=ctx.room, agent=agent)
    agent.set_session(session)  # Provide session reference for TTS output

    # Generate initial greeting
    await session.generate_reply(instructions="Say 'non-verbal cues description started'.")

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


