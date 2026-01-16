//
// AudioManager.swift
//
// Manages audio playback and routing to Meta Wearables (Ray-Ban Meta glasses).
// Configures AVAudioSession for Bluetooth audio and provides utility methods
// for playing test sounds with left/right panning.
//

import AVFoundation

enum AudioChannel {
  case left
  case right
  case center
}

enum AudioManagerError: Error {
  case audioSessionConfigurationFailed
  case soundGenerationFailed
  case playbackFailed
}

@MainActor
class AudioManager: ObservableObject {
  static let shared = AudioManager()

  @Published var isAudioSessionConfigured = false
  @Published var availableAudioRoutes: [String] = []

  private var audioPlayer: AVAudioPlayer?

  private init() {
    setupNotifications()
  }

  /// Configure audio session for Bluetooth audio routing to Meta Wearables
  /// - Parameter enableRecording: If true, configures for duplex audio (recording + playback) for LiveKit
  func configureAudioSession(enableRecording: Bool = false) throws {
    let audioSession = AVAudioSession.sharedInstance()

    do {
      if enableRecording {
        // For LiveKit: Use playAndRecord category for simultaneous mic input and speaker output
        // This allows microphone publishing to agent AND receiving agent speech
        try audioSession.setCategory(
          .playAndRecord,
          mode: .voiceChat,  // Optimized for voice communication
          options: [.allowBluetooth, .allowBluetoothA2DP]  // Enable Bluetooth routing
        )
        NSLog("[AudioManager] Audio session configured for duplex audio (recording + playback)")
      } else {
        // For simple playback: Use playback category
        // Playback category automatically routes to Bluetooth devices (including Meta Wearables)
        try audioSession.setCategory(
          .playback,
          mode: .default
        )
        NSLog("[AudioManager] Audio session configured for playback only")
      }

      // Activate the audio session
      try audioSession.setActive(true)

      isAudioSessionConfigured = true
      updateAvailableRoutes()

      NSLog("[AudioManager] Audio session configured successfully")
      NSLog("[AudioManager] Current route: \(audioSession.currentRoute.outputs.map { $0.portName }.joined(separator: ", "))")
    } catch {
      NSLog("[AudioManager] Failed to configure audio session: \(error)")
      throw AudioManagerError.audioSessionConfigurationFailed
    }
  }

  /// Get current audio output route
  func getCurrentAudioRoute() -> String {
    let audioSession = AVAudioSession.sharedInstance()
    return audioSession.currentRoute.outputs.map { $0.portName }.joined(separator: ", ")
  }

  /// Update available audio routes
  private func updateAvailableRoutes() {
    let audioSession = AVAudioSession.sharedInstance()
    availableAudioRoutes = audioSession.currentRoute.outputs.map { output in
      "\(output.portName) (\(output.portType.rawValue))"
    }
  }

  /// Setup notifications for audio route changes
  private func setupNotifications() {
    NotificationCenter.default.addObserver(
      forName: AVAudioSession.routeChangeNotification,
      object: nil,
      queue: .main
    ) { [weak self] notification in
      self?.handleRouteChange(notification)
    }
  }

  /// Handle audio route changes (e.g., glasses connected/disconnected)
  private func handleRouteChange(_ notification: Notification) {
    guard let userInfo = notification.userInfo,
          let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
          let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else {
      return
    }

    updateAvailableRoutes()

    switch reason {
    case .newDeviceAvailable:
      NSLog("[AudioManager] New audio device connected")
    case .oldDeviceUnavailable:
      NSLog("[AudioManager] Audio device disconnected")
    default:
      NSLog("[AudioManager] Audio route changed: \(reason)")
    }
  }

  /// Play a ping sound with specified panning (left/right ear)
  func playPing(channel: AudioChannel, frequency: Double = 800, duration: Double = 0.2) throws {
    // Generate ping sound
    guard let audioData = generatePingSound(frequency: frequency, duration: duration) else {
      throw AudioManagerError.soundGenerationFailed
    }

    // Create temporary file for the audio
    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("ping.wav")
    try audioData.write(to: tempURL)

    // Create audio player
    audioPlayer = try AVAudioPlayer(contentsOf: tempURL)
    guard let player = audioPlayer else {
      throw AudioManagerError.playbackFailed
    }

    // Set panning based on channel
    switch channel {
    case .left:
      player.pan = -1.0  // Full left
    case .right:
      player.pan = 1.0   // Full right
    case .center:
      player.pan = 0.0   // Center
    }

    player.prepareToPlay()
    player.play()

    NSLog("[AudioManager] Playing ping on \(channel) channel")
  }

  /// Generate a simple sine wave ping sound
  private func generatePingSound(frequency: Double, duration: Double) -> Data? {
    let sampleRate: Double = 44100
    let amplitude: Double = 0.3
    let samples = Int(sampleRate * duration)

    var audioData = Data()

    // WAV file header
    let wavHeader = createWAVHeader(sampleRate: Int(sampleRate), samples: samples)
    audioData.append(wavHeader)

    // Generate sine wave samples
    for i in 0..<samples {
      let time = Double(i) / sampleRate
      let value = sin(2.0 * .pi * frequency * time) * amplitude

      // Apply fade out to avoid clicks
      let fadeOut = i > samples - 1000 ? Double(samples - i) / 1000.0 : 1.0
      let finalValue = value * fadeOut

      // Convert to 16-bit PCM
      let sample = Int16(finalValue * 32767)
      audioData.append(Data(bytes: [UInt8(sample & 0xFF), UInt8((sample >> 8) & 0xFF)], count: 2))
    }

    return audioData
  }

  /// Create WAV file header
  private func createWAVHeader(sampleRate: Int, samples: Int) -> Data {
    var header = Data()
    let byteRate = sampleRate * 2  // 16-bit mono
    let dataSize = samples * 2

    // RIFF header
    header.append("RIFF".data(using: .ascii)!)
    header.append(Data(bytes: [(UInt32(dataSize + 36) & 0xFF), ((UInt32(dataSize + 36) >> 8) & 0xFF), ((UInt32(dataSize + 36) >> 16) & 0xFF), ((UInt32(dataSize + 36) >> 24) & 0xFF)].map { UInt8($0) }, count: 4))
    header.append("WAVE".data(using: .ascii)!)

    // Format chunk
    header.append("fmt ".data(using: .ascii)!)
    header.append(Data([16, 0, 0, 0]))  // Chunk size
    header.append(Data([1, 0]))  // PCM format
    header.append(Data([1, 0]))  // Mono
    header.append(Data(bytes: [(UInt32(sampleRate) & 0xFF), ((UInt32(sampleRate) >> 8) & 0xFF), ((UInt32(sampleRate) >> 16) & 0xFF), ((UInt32(sampleRate) >> 24) & 0xFF)].map { UInt8($0) }, count: 4))
    header.append(Data(bytes: [(UInt32(byteRate) & 0xFF), ((UInt32(byteRate) >> 8) & 0xFF), ((UInt32(byteRate) >> 16) & 0xFF), ((UInt32(byteRate) >> 24) & 0xFF)].map { UInt8($0) }, count: 4))
    header.append(Data([2, 0]))  // Block align
    header.append(Data([16, 0]))  // Bits per sample

    // Data chunk
    header.append("data".data(using: .ascii)!)
    header.append(Data(bytes: [(UInt32(dataSize) & 0xFF), ((UInt32(dataSize) >> 8) & 0xFF), ((UInt32(dataSize) >> 16) & 0xFF), ((UInt32(dataSize) >> 24) & 0xFF)].map { UInt8($0) }, count: 4))

    return header
  }

  /// Stop any currently playing audio
  func stopPlayback() {
    audioPlayer?.stop()
    audioPlayer = nil
  }

  deinit {
    NotificationCenter.default.removeObserver(self)
  }
}
