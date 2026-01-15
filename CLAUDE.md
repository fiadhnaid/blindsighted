# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Blindsighted is a mobile app with FastAPI backend that provides AI-powered visual assistance for blind/visually impaired users using Meta AI Glasses (Ray-Ban Meta).

**Architecture**: Monorepo with two main components:
- `ios/` - Native iOS app (Swift/SwiftUI) using Meta Wearables DAT SDK for Ray-Ban Meta glasses
- `api/` - FastAPI backend (Python 3.11) that processes frames using Gemini vision and ElevenLabs TTS

**Flow**: Glasses capture photo → App sends base64 image to API → Gemini describes scene → ElevenLabs converts to speech → Audio played to user

## Development Commands

### iOS App (Swift/SwiftUI)

```bash
cd ios
open Blindsighted.xcodeproj     # Open in Xcode

# Build and run on simulator
xcodebuild -project Blindsighted.xcodeproj \
  -scheme Blindsighted \
  -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 15 Pro'

# Build for device
xcodebuild -project Blindsighted.xcodeproj \
  -scheme Blindsighted \
  -sdk iphoneos
```

**Dependencies**:
- Meta Wearables DAT SDK (integrated via Swift Package Manager in Xcode)
  - `MWDATCore` - Core wearables functionality
  - `MWDATCamera` - Camera streaming and photo capture
  - Package URL: `https://github.com/facebook/meta-wearables-dat-ios`
  - Version: 0.3.0

**Note**: The app is based on Meta's CameraAccess sample from the DAT SDK, customized for Blindsighted.

**Adding New Files to Xcode Project**:

When creating new Swift files outside of Xcode, you must add them to the project file:

```bash
# Install xcodeproj gem (one time)
gem install xcodeproj

# Add a new file to the project
cd ios
xcodeproj --project Blindsighted.xcodeproj add-file Blindsighted/path/to/File.swift --group GroupName
```

**Common group names:**
- `Utils` - For utility files like VideoRecorder.swift
- `ViewModels` - For view model files
- `Views` - For SwiftUI view files
- `Views/Components` - For reusable UI components

**Example:**
```bash
# Add a new utility file
xcodeproj --project Blindsighted.xcodeproj add-file Blindsighted/Utils/AudioRecorder.swift --group Utils

# Add a new view model
xcodeproj --project Blindsighted.xcodeproj add-file Blindsighted/ViewModels/SettingsViewModel.swift --group ViewModels

# Add a new view
xcodeproj --project Blindsighted.xcodeproj add-file Blindsighted/Views/SettingsView.swift --group Views
```

**Important**: Always add new files to the Xcode project, or the build will fail with "file not found" errors.

### API (FastAPI/Python)

```bash
cd api
uv pip install -e ".[dev]"     # Install with dev dependencies
uv pip install -e .            # Install production only
uvicorn main:app --reload --host 0.0.0.0 --port 8000  # Run dev server
ruff check --fix .             # Lint and auto-fix
ruff format .                  # Format code
mypy .                         # Type check
```

**Configuration**: Copy `api/.env.example` to `api/.env` and add:
- `OPENROUTER_API_KEY` - Get from https://openrouter.ai/
- `ELEVENLABS_API_KEY` - Get from https://elevenlabs.io/

### Docker

```bash
cd api
docker build -t blindsighted-api .
docker run -p 8000:8000 --env-file .env blindsighted-api
```

## Code Architecture

### Dependency Injection Pattern (API)

The FastAPI backend uses dependency injection via `Annotated` types. Do NOT create global client instances.

**Correct**:
```python
from typing import Annotated
from fastapi import Depends

def get_client() -> Client:
    return Client()

@app.post("/endpoint")
async def endpoint(client: Annotated[Client, Depends(get_client)]):
    await client.do_something()
```

**Incorrect** (DON'T DO THIS):
```python
# Do not create global instances
global_client = Client()  # ❌ Wrong
```

See `api/main.py:22-30` for examples.

### Configuration Management

- **API**: Uses `pydantic-settings` to load from `.env` files. See `api/config.py`.
- **iOS App**: Configuration is managed via Info.plist and app entitlements. No external config files needed for the iOS app itself.

## CI/CD & Releases

### iOS Build

iOS builds are performed using Xcode and can be distributed via:
- **Development**: Build directly from Xcode to physical device or simulator
- **TestFlight**: Archive and upload to App Store Connect for beta testing
- **App Store**: Production releases via App Store Connect

**Creating an Archive**:
1. In Xcode: Product → Archive
2. Window → Organizer to manage archives
3. Distribute App → choose distribution method

### GitHub Actions Workflows

- **PR Checks** (`.github/workflows/pr-checks.yml`): Lint/format (ruff), type check (mypy) for API
- **Release** (`.github/workflows/release.yml`): Triggered on `v*.*.*` tags
  - Builds Docker image for API and pushes to `ghcr.io/djrhails/blindsighted/api`
  - Creates GitHub release with changelog

**Creating a Release**:
```bash
git tag v1.2.3
git push origin v1.2.3
```

### Package Manager

- **iOS App**: Swift Package Manager (integrated in Xcode)
- **API**: Uses `uv` for Python dependency management

## Python Code Style

- **Line length**: 100 characters (ruff config)
- **Type hints**: Strict mode enabled, all functions must have type hints
- **Imports**: Auto-sorted by ruff (isort)
- **Python version**: 3.11+ required

## iOS App Architecture

- **UI Framework**: SwiftUI with declarative views
- **State Management**: SwiftUI's `@StateObject`, `@ObservedObject`, and `@Published` properties
- **Architecture Pattern**: MVVM (Model-View-ViewModel)
  - **Views**: SwiftUI views in `ios/Blindsighted/Views/`
  - **ViewModels**: Observable objects in `ios/Blindsighted/ViewModels/`
  - **Models**: Data models from Meta Wearables DAT SDK

### Key Components

- **WearablesViewModel** (`ios/Blindsighted/ViewModels/WearablesViewModel.swift`): Manages device connection and registration
- **StreamSessionViewModel** (`ios/Blindsighted/ViewModels/StreamSessionViewModel.swift`): Handles video streaming, photo capture, and session state
- **Meta Wearables DAT SDK Integration**:
  - SDK configured once at app launch in `BlindsightedApp.swift`
  - Listener pattern for SDK events (state changes, video frames, errors)
  - `StreamSession` manages streaming lifecycle

### Video Streaming Flow

1. User taps "Start Streaming" → requests camera permission
2. `StreamSession.start()` initiates connection to glasses
3. Video frames received via `videoFramePublisher` listener
4. Frames converted to `UIImage` and displayed in real-time
5. User can capture photos during stream with `capturePhoto()`

## Troubleshooting

### iOS Build Requirements

- **Xcode**: 26.2+
- **Swift**: 6.2+
- **iOS Deployment Target**: 17.0+

The project is configured for:
- Swift version: 6.2 (in `ios/Blindsighted.xcodeproj/project.pbxproj`)
- iOS deployment target: 17.0 (matches Meta Wearables SDK requirement)

### Meta Wearables SDK Package Not Found

If you see errors like `Missing package product 'MWDATCore'` or `Missing package product 'MWDATCamera'`:

**Problem**: Swift Package Manager may not automatically resolve packages.

**Solution 1: Resolve in Xcode** (Recommended)
1. Open `ios/Blindsighted.xcodeproj` in Xcode
2. Go to **File → Packages → Resolve Package Versions**
3. Wait for resolution to complete
4. Clean build folder: **Product → Clean Build Folder** (⇧⌘K)
5. Build the project

**Solution 2: Manually Add SPM Dependency**
If automatic resolution fails, manually add the package:

1. Open `ios/Blindsighted.xcodeproj` in Xcode
2. Select the **Blindsighted** project in Project Navigator
3. Select the **Blindsighted** target
4. Go to **General** tab → **Frameworks, Libraries, and Embedded Content**
5. Click **+** → **Add Package Dependency**
6. Enter: `https://github.com/facebook/meta-wearables-dat-ios`
7. Set version: **Exact Version 0.3.0**
8. Select products: **MWDATCore** and **MWDATCamera**
9. Clean and rebuild

**Solution 3: Clear Derived Data**
```bash
rm -rf ~/Library/Developer/Xcode/DerivedData/Blindsighted-*
```

Then open Xcode and use Solution 1 or 2.

### Swift Version Mismatch

If you see Swift version errors:

1. Open `ios/Blindsighted.xcodeproj` in Xcode
2. Select the **Blindsighted** project in Project Navigator
3. Select the **Blindsighted** target
4. Go to **Build Settings** tab
5. Search for "Swift Language Version"
6. Ensure it's set to **Swift 6.2** (or 6.0+)
7. Clean build folder: **Product → Clean Build Folder** (⇧⌘K)
8. Rebuild the project
