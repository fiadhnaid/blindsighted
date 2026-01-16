# Blindsighted (Sample App)

**A hackathon-ready template for building AI-powered experiences with Ray-Ban Meta smart glasses.**

Blindsighted is a **sample app** that connects Ray-Ban Meta smart glasses to AI agents via LiveKit. The context is for a visual assistance app for blind/visually impaired users, but the architecture works for any AI-powered glasses experience.

The integration setup with Meta's wearables SDK and LiveKit streaming was finicky to get right. This template gives you a working foundation so you can skip that part and jump straight to the interesting bits.

## Architecture Overview

```
iOS App (Swift) → LiveKit Cloud (WebRTC) → AI Agents (Python)
     ↓                                           ↑
     └──────→ FastAPI Backend (optional) ───────┘
              (sessions, storage, etc.)
```

**Three independent components:**

- **`ios/`** - Native iOS app using Meta Wearables DAT SDK

  - Streams video/audio from Ray-Ban Meta glasses to LiveKit
  - Receives audio/data responses from agents
  - Works standalone if you just want to test the glasses SDK

- **`agents/`** - LiveKit agents (Python)

  - Join LiveKit rooms as peers
  - Process live video/audio streams with AI models
  - Send responses back via audio/video/data channels
  - **This is where the magic happens** - build your AI features here

- **`api/`** - FastAPI backend (Python)
  - Session management and room creation
  - R2 storage for life logs and replays
  - Optional but useful for anything ad hoc you need a backend for

**You can work on just one part.** Want to build a cool agent but not touch iOS? Great. Want to experiment with the glasses SDK without running agents? Also fine. Want to add interesting storage/indexing features? The backend's there for you.

## Quick Start

### iOS App

```bash
cd ios
open Blindsighted.xcodeproj
# Build and run in Xcode (⌘R)
```

**Requirements**: Xcode 26.2+, iOS 17.0+, Swift 6.2+

See [ios/README.md](ios/README.md) for detailed setup.

### Agents

```bash
cd agents
uv sync
uv run example_agent.py dev
```

**Test without hardware**: Use the [LiveKit Agents Playground](https://agents-playground.livekit.io/) to test agents with your webcam/microphone instead of glasses.

See [agents/README.md](agents/README.md) for agent development.

### API Backend (Optional)

```bash
cd api
uv sync
uv run main.py
```

API docs at `http://localhost:8000/docs`

## What's Included

### iOS App Features

- Live video streaming from Ray-Ban Meta glasses
- Audio routing to/from glasses (left/right channel testing)
- Photo capture during streaming
- Video recording and local storage
- Video gallery with playback
- LiveKit integration with WebRTC
- Share videos/photos

### Agent Template

- LiveKit room auto-join based on session
- Audio/video stream processing
- AI model integration examples (vision, TTS)
- Bidirectional communication (receive video, send audio)

### Backend API

- Session management endpoints
- LiveKit room creation with tokens
- R2 storage integration for life logs
- FastAPI with dependency injection patterns

## Use It Your Way

**Feel free to:**

- Rip out everything you don't need
- Replace the AI models with your own
- Change the entire agent architecture
- Use a different backend (or no backend)
- Build something completely different on top of the glasses SDK

**This is over-engineered for a hackathon.** The three-component architecture exists because I found the initial setup painful and wanted to provide options. If you have a better approach or this feels too complicated, throw it away! The point is to give you working examples to learn from, not to force an architecture on you.

## Environment Variables & API Keys

The app needs a few API keys to work:

- **LiveKit**: Server URL, API key, API secret (for WebRTC streaming)
- **OpenRouter API Key** (optional, for AI models)
- **ElevenLabs API Key** (optional, for TTS)

**Having trouble getting something running?** Reach out and I'll unblock you.

See `ios/Config.xcconfig.example` and `api/.env.example` for configuration details.

## Documentation

- **CLAUDE.md** - Full development guide with architecture details, code patterns, troubleshooting
- **ios/README.md** - iOS-specific setup and configuration
- **agents/README.md** - Agent development guide
- **api/** - Backend API with OpenAPI docs at `/docs`

## License

**In short:** Keep it open source, it's fine to make money with it. I'd love to see what you build with it.

**Exception**: The iOS app incorporates sample code from Meta's [meta-wearables-dat-ios](https://github.com/facebook/meta-wearables-dat-ios) repository, which has its own license terms. Check that repo for Meta's SDK license.

## Why Does This Exist?

I built this because:

1. Getting Meta's wearables SDK working took a bit of time without being 'fun'.
2. Originally I had custom WebRTC streaming (which took a lot of time); Pentaform showed me LiveKit which seems much more suitable for a hackathon use-case so I swapped over to that for this project, but also has it's own pain points.
3. Unlikely typical hackathons which are one-and-done, it'd be great to have something people can iterate on.

If this helps you build something cool, that's awesome. If you find a better way to do any of this, even better.

## Contributing

Found a bug? Have a better pattern? PRs welcome. This is meant to help people, so improvements that make it easier to use or understand are great.
