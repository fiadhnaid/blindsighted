# Blindsighted Agents

**Part of the Blindsighted hackathon template** - this is where you build your AI features.

These are self-contained LiveKit agents that process video/audio streams from Ray-Ban Meta glasses. The example agent does vision-based assistance, but you can customize these agents to do anything you want.

## What This Component Does

- Joins LiveKit rooms as peers (alongside the iOS app)
- Receives video/audio streams from iOS app via LiveKit
- Processes through AI models (STT → LLM → TTS pipeline)
- Publishes audio/data responses back to iOS app
- Can optionally log conversation turns to backend API

**This is the most important part** - all your AI logic goes here. The iOS app and backend are just plumbing to get data to/from your agents.

**See [../ARCHITECTURE.md](../ARCHITECTURE.md) for how this fits into the overall system.**

## Architecture

Agents are **completely independent** from the API backend:

- **`api/`** - Infrastructure (room management, tokens, database) - _optional_
- **`agents/`** - Custom AI logic (STT, LLM, TTS, vision processing) - _this is where you innovate_

You can run agents without the backend API. Just get LiveKit credentials and you're good to go.

## Vision Agent

The vision agent uses a streaming STT-LLM-TTS pipeline with video frame analysis:

**Pipeline:**

1. **STT** (Deepgram) - Converts user speech to text
2. **LLM** (Gemini 2.5 Flash) - Processes text + video frames, generates responses
3. **TTS** (ElevenLabs) - Converts responses to natural speech

**Vision Integration:**

- Buffers the latest video frame from user's camera
- Attaches frame to each conversation turn
- LLM can see what the user sees and describe the environment

## Setup

### 1. Install Dependencies

```bash
cd agents
uv pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Required configuration:

- **LiveKit** - Already configured (from API setup)
  - `LIVEKIT_API_KEY` - Your LiveKit API key
  - `LIVEKIT_API_SECRET` - Your LiveKit API secret
  - `LIVEKIT_URL` - Your LiveKit server URL (e.g., `wss://your-project.livekit.cloud`)
  - `LIVEKIT_AGENT_NAME` - Agent identifier for filtering so people can re-use the same livekit project (e.g., `vision-agent`)
- **OpenRouter** - Get from https://openrouter.ai/
  - `OPENROUTER_API_KEY` - For Gemini vision model access
- **ElevenLabs** - Get from https://elevenlabs.io/
  - `ELEVENLABS_API_KEY` - For text-to-speech
  - `ELEVENLABS_VOICE_ID` - Voice ID (default: Rachel)
- **Deepgram** - Get from https://deepgram.com/
  - `DEEPGRAM_API_KEY` - For speech-to-text

### 3. Run the Agent

```bash
# Development mode
uv run python vision_agent.py dev

# Production mode
uv run python vision_agent.py start
```

The agent will:

1. Connect to LiveKit Cloud
2. Wait for rooms to be created (via API `/sessions/start`)
3. Join rooms as a participant
4. Listen for user speech and video
5. Respond with scene descriptions and assistance

## Testing Without Hardware

### LiveKit Agents Playground

You can test your agents without the iOS app or Ray-Ban Meta glasses using the **LiveKit Agents Playground**: https://agents-playground.livekit.io/

This web-based tool provides a browser interface for interacting with your agents via voice and video, perfect for rapid development and debugging.

**When to use it:**

- Developing and testing agent logic without hardware
- Iterating on prompts and AI model configurations
- Debugging audio/video processing issues
- Testing different AI model combinations
- Demonstrating agent capabilities without physical devices

### How to Use the Playground

**1. Start your agent locally:**

```bash
cd agents
uv run python vision_agent.py dev
```

The agent will connect to your LiveKit Cloud instance and wait for rooms.

**2. Open the Agents Playground:**

Visit https://agents-playground.livekit.io/

**3. Configure connection:**

- **Server URL**: Your LiveKit server URL (e.g., `wss://your-project.livekit.cloud`)
- **API Key**: Your LiveKit API key (from `.env`)
- **API Secret**: Your LiveKit API secret (from `.env`)

**4. Connect and interact:**

- Click **"Connect"** to join a room
- The playground will automatically create a room and your agent will join
- Grant microphone and camera permissions
- Start speaking - your agent will respond!

### Playground Features

**Audio Input:**
- Speak directly into your microphone
- Agent processes speech via STT pipeline
- Responses play through your speakers

**Video Input:**
- Share your webcam for vision-enabled agents
- Agent receives video frames just like from glasses
- Test scene description and visual understanding

**Debug Panel:**
- View real-time transcriptions
- See agent responses before TTS
- Monitor connection status and tracks
- Check latency metrics

### Testing Vision Features

For the vision agent, the playground is excellent for testing:

```bash
# Start vision agent
uv run python vision_agent.py dev

# In playground:
# 1. Enable your webcam
# 2. Ask: "What do you see?"
# 3. Agent describes your webcam view
```

The agent processes webcam frames exactly like it would process glasses camera frames, so you can iterate on vision prompts and logic without hardware.

### Agent Filtering with Playground

If you're using agent filtering (see [Agent Filtering System](#agent-filtering-system)), you'll need to:

**Option 1: Use default agent (no filtering)**
- Don't set `LIVEKIT_AGENT_NAME` in `.env`
- Agent accepts all rooms (backward compatibility)

**Option 2: Create rooms with metadata via API**
- Keep using the backend API to create rooms with `agent_id`
- Agent only joins rooms with matching metadata
- Playground connects to pre-existing rooms

**Option 3: Temporarily disable filtering for testing**
- Comment out the `request_fnc` parameter:
  ```python
  # @server.rtc_session(request_fnc=should_accept_job)  # Temporarily disabled
  @server.rtc_session()
  async def vision_agent(ctx: JobContext) -> None:
      ...
  ```

### Limitations

The playground is for development/testing only:

- No persistent storage (sessions not saved to database)
- No segment logging to backend
- Room names are auto-generated
- Requires exposing LiveKit credentials in browser

For production testing with the full stack (iOS app + glasses + backend), see the main setup guide.

## Customization

### Create Your Own Agent

Copy `vision_agent.py` and modify:

**Change the AI models:**

```python
session = AgentSession(
    stt="assemblyai/universal-streaming",  # Different STT
    llm=openai.LLM(model="gpt-4o"),        # Different LLM
    tts="cartesia/sonic-3",                 # Different TTS
    vad=silero.VAD.load(),
)
```

**Customize instructions:**

```python
class CustomAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""Your custom system prompt here.
            Describe how the agent should behave."""
        )
```

**Add custom logic:**

```python
async def on_user_turn_completed(self, chat_ctx, new_message):
    # Custom processing before LLM
    if self._latest_frame:
        # Analyze frame, add metadata, etc.
        new_message.content.append(llm.ChatImage(image=self._latest_frame))
```

**Add agent filtering:**

```python
import os

def should_accept_job(job_request) -> bool:
    """Filter jobs by agent name."""
    agent_name = os.getenv("LIVEKIT_AGENT_NAME", "my-agent")
    room_metadata = job_request.room.metadata

    # Accept if metadata matches or is empty (backward compatibility)
    return not room_metadata or room_metadata == agent_name

@server.rtc_session(request_fnc=should_accept_job)
async def my_custom_agent(ctx: JobContext) -> None:
    # Your agent logic here
    session = AgentSession(...)
    await session.start(room=ctx.room, agent=CustomAssistant())
```

### Supported Models

**STT (Speech-to-Text):**

- `deepgram/nova-2` - Fast, accurate
- `assemblyai/universal-streaming` - Multilingual
- See [LiveKit STT docs](https://docs.livekit.io/agents/models/stt/)

**LLM (Language Models):**

- `google/gemini-2.0-flash-exp:free` - Vision support via OpenRouter
- `openai/gpt-4o` - OpenAI multimodal
- `anthropic/claude-sonnet-4.5` - Anthropic via OpenRouter
- See [LiveKit LLM docs](https://docs.livekit.io/agents/models/llm/)

**TTS (Text-to-Speech):**

- `elevenlabs` - Natural, expressive voices
- `cartesia/sonic-3` - Fast, low latency
- `openai/tts-1` - OpenAI voices
- See [LiveKit TTS docs](https://docs.livekit.io/agents/models/tts/)

## Agent Filtering System

The agent filtering system allows you to run multiple specialized agents simultaneously, with each agent only handling sessions intended for it.

### How It Works

1. **API stores agent ID** - When creating a session, the API stores the `agent_id` in the LiveKit room metadata
2. **Agent filters jobs** - Each agent worker reads `LIVEKIT_AGENT_NAME` from its environment and only accepts rooms where the metadata matches
3. **Multiple agents** - You can run different agents (vision, transcription, etc.) and route sessions to specific agents

### Configure Agent Name

Set the agent name in your `.env` file:

```bash
# .env
LIVEKIT_AGENT_NAME=vision-agent
```

The agent will only accept jobs for rooms created with a matching `agent_id`.

### Start Session with Agent ID

When creating a session via the API, specify which agent should handle it:

```bash
curl -X POST http://localhost:8000/sessions/start \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "device_id": "device456",
    "agent_id": "vision-agent"
  }'
```

The agent worker with `LIVEKIT_AGENT_NAME=vision-agent` will accept this job.

### Example: Multiple Agents

Run different specialized agents simultaneously:

```bash
# Terminal 1 - Vision agent
cd agents
LIVEKIT_AGENT_NAME=vision-agent uv run python vision_agent.py dev

# Terminal 2 - Transcription agent (hypothetical)
cd agents
LIVEKIT_AGENT_NAME=transcription-agent uv run python transcription_agent.py dev
```

Then route sessions to specific agents:

```bash
# Route to vision agent
curl -X POST http://localhost:8000/sessions/start \
  -d '{"agent_id": "vision-agent"}'

# Route to transcription agent
curl -X POST http://localhost:8000/sessions/start \
  -d '{"agent_id": "transcription-agent"}'
```

### Backward Compatibility

If no `agent_id` is specified when creating a session (room metadata is empty), agents will accept the job for backward compatibility. A warning will be logged:

```
Room blindsighted-xyz has no metadata - accepting job for backward compatibility
```

## Replay Functionality

The `segments` table in the database tracks conversation turns for replay:

**Capture segments** - Log each turn with timestamps and metadata
**Replay sessions** - Process stored segments with a different AI agent

This allows experimenting with different models on the same conversation.

## Deployment

### Development

```bash
uv run python vision_agent.py dev
```

### Production

```bash
# Run as a service
uv run python vision_agent.py start

# Or with systemd, Docker, etc.
```

### Multiple Agents

You can run multiple agent workers simultaneously. Use `LIVEKIT_AGENT_NAME` to ensure each agent only handles its designated sessions:

```bash
# Terminal 1 - Vision agent
LIVEKIT_AGENT_NAME=vision-agent uv run python vision_agent.py dev

# Terminal 2 - Custom agent with different name
LIVEKIT_AGENT_NAME=custom-agent uv run python custom_agent.py dev
```

Each agent worker can handle multiple rooms concurrently. See the [Agent Filtering System](#agent-filtering-system) section for details on routing sessions to specific agents.

## Troubleshooting

**Agent not joining rooms:**

- Check LiveKit credentials in `.env`
- Verify agent can connect: `uv run python vision_agent.py dev`
- Check LiveKit Cloud dashboard for connected agents
- **Agent filtering issue**: Verify `LIVEKIT_AGENT_NAME` in agent's `.env` matches the `agent_id` used in `/sessions/start` request
- Check agent logs for "Rejecting job" messages indicating a mismatch

**No video frames:**

- Ensure user grants camera permission in iOS app
- Check video track is published: LiveKit dashboard > Room > Tracks
- Verify agent subscribed: Check logs for "Subscribed to existing video track"

**TTS not working:**

- Verify ElevenLabs API key in `.env`
- Check API quota/limits
- Try different TTS provider (Cartesia, OpenAI)

**LLM errors:**

- Verify OpenRouter API key for Gemini
- Check model name is correct
- Try different model (GPT-4, Claude)

## Resources

- [LiveKit Agents Docs](https://docs.livekit.io/agents/)
- [LiveKit Python SDK](https://github.com/livekit/python-sdks)
- [Vision Agent Example](https://docs.livekit.io/agents/build/vision)
