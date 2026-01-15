//
// AudioTestView.swift
//
// Test view for verifying audio routing to Meta Wearables.
// Allows testing left/right ear audio panning with ping sounds.
//

import SwiftUI

struct AudioTestView: View {
  @StateObject private var audioManager = AudioManager.shared
  @State private var showError = false
  @State private var errorMessage = ""

  var body: some View {
    ScrollView {
      VStack(spacing: 24) {
        // Header
        VStack(spacing: 8) {
          Image(systemName: "headphones")
            .font(.system(size: 48))
            .foregroundColor(.blue)

          Text("Audio Test")
            .font(.system(size: 24, weight: .bold))

          Text("Test audio routing to your Meta glasses")
            .font(.system(size: 15))
            .foregroundColor(.secondary)
            .multilineTextAlignment(.center)
        }
        .padding(.top, 24)

        // Audio Session Status
        VStack(alignment: .leading, spacing: 12) {
          Label("Audio Session", systemImage: "waveform")
            .font(.system(size: 16, weight: .semibold))

          HStack {
            Circle()
              .fill(audioManager.isAudioSessionConfigured ? Color.green : Color.gray)
              .frame(width: 12, height: 12)

            Text(audioManager.isAudioSessionConfigured ? "Configured" : "Not Configured")
              .font(.system(size: 14))
              .foregroundColor(.secondary)

            Spacer()

            if !audioManager.isAudioSessionConfigured {
              Button("Configure") {
                configureAudioSession()
              }
              .font(.system(size: 14, weight: .medium))
            }
          }
          .padding(12)
          .background(Color.gray.opacity(0.1))
          .cornerRadius(8)
        }
        .padding(.horizontal, 24)

        // Current Audio Route
        if audioManager.isAudioSessionConfigured {
          VStack(alignment: .leading, spacing: 12) {
            Label("Current Audio Route", systemImage: "speaker.wave.2")
              .font(.system(size: 16, weight: .semibold))

            VStack(alignment: .leading, spacing: 8) {
              Text(audioManager.getCurrentAudioRoute())
                .font(.system(size: 14, weight: .medium))

              if !audioManager.availableAudioRoutes.isEmpty {
                ForEach(audioManager.availableAudioRoutes, id: \.self) { route in
                  Text(route)
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
                }
              }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(Color.blue.opacity(0.1))
            .cornerRadius(8)
          }
          .padding(.horizontal, 24)
        }

        // Ear Test Controls
        VStack(spacing: 16) {
          Text("Test Audio Channels")
            .font(.system(size: 16, weight: .semibold))

          // Left Ear
          HStack(spacing: 16) {
            Image(systemName: "l.square.fill")
              .font(.system(size: 32))
              .foregroundColor(.blue)

            VStack(alignment: .leading, spacing: 4) {
              Text("Left Ear")
                .font(.system(size: 16, weight: .semibold))
              Text("Test left audio channel")
                .font(.system(size: 13))
                .foregroundColor(.secondary)
            }

            Spacer()

            Button(action: {
              playPing(channel: .left)
            }) {
              Image(systemName: "play.circle.fill")
                .font(.system(size: 36))
                .foregroundColor(.blue)
            }
            .disabled(!audioManager.isAudioSessionConfigured)
          }
          .padding(16)
          .background(Color.gray.opacity(0.05))
          .cornerRadius(12)

          // Right Ear
          HStack(spacing: 16) {
            Image(systemName: "r.square.fill")
              .font(.system(size: 32))
              .foregroundColor(.orange)

            VStack(alignment: .leading, spacing: 4) {
              Text("Right Ear")
                .font(.system(size: 16, weight: .semibold))
              Text("Test right audio channel")
                .font(.system(size: 13))
                .foregroundColor(.secondary)
            }

            Spacer()

            Button(action: {
              playPing(channel: .right)
            }) {
              Image(systemName: "play.circle.fill")
                .font(.system(size: 36))
                .foregroundColor(.orange)
            }
            .disabled(!audioManager.isAudioSessionConfigured)
          }
          .padding(16)
          .background(Color.gray.opacity(0.05))
          .cornerRadius(12)

          // Center/Both
          HStack(spacing: 16) {
            Image(systemName: "speaker.wave.3.fill")
              .font(.system(size: 32))
              .foregroundColor(.green)

            VStack(alignment: .leading, spacing: 4) {
              Text("Center (Both)")
                .font(.system(size: 16, weight: .semibold))
              Text("Test both channels")
                .font(.system(size: 13))
                .foregroundColor(.secondary)
            }

            Spacer()

            Button(action: {
              playPing(channel: .center)
            }) {
              Image(systemName: "play.circle.fill")
                .font(.system(size: 36))
                .foregroundColor(.green)
            }
            .disabled(!audioManager.isAudioSessionConfigured)
          }
          .padding(16)
          .background(Color.gray.opacity(0.05))
          .cornerRadius(12)
        }
        .padding(.horizontal, 24)

        // Help Text
        VStack(alignment: .leading, spacing: 12) {
          Label("How to use", systemImage: "info.circle")
            .font(.system(size: 16, weight: .semibold))

          VStack(alignment: .leading, spacing: 8) {
            HelpItemView(
              number: "1",
              text: "Connect your Meta glasses via Bluetooth"
            )
            HelpItemView(
              number: "2",
              text: "Tap 'Configure' to set up audio routing"
            )
            HelpItemView(
              number: "3",
              text: "Use the play buttons to test each ear"
            )
            HelpItemView(
              number: "4",
              text: "Verify you hear the ping in the correct ear"
            )
          }
        }
        .padding(.horizontal, 24)
        .padding(.bottom, 24)
      }
    }
    .alert("Error", isPresented: $showError) {
      Button("OK") {
        showError = false
      }
    } message: {
      Text(errorMessage)
    }
    .onAppear {
      // Auto-configure on appear if not already configured
      if !audioManager.isAudioSessionConfigured {
        configureAudioSession()
      }
    }
  }

  private func configureAudioSession() {
    do {
      try audioManager.configureAudioSession()
    } catch {
      errorMessage = "Failed to configure audio session: \(error.localizedDescription)"
      showError = true
    }
  }

  private func playPing(channel: AudioChannel) {
    do {
      try audioManager.playPing(channel: channel)
    } catch {
      errorMessage = "Failed to play audio: \(error.localizedDescription)"
      showError = true
    }
  }
}

struct HelpItemView: View {
  let number: String
  let text: String

  var body: some View {
    HStack(alignment: .top, spacing: 12) {
      Text(number)
        .font(.system(size: 14, weight: .bold))
        .foregroundColor(.white)
        .frame(width: 24, height: 24)
        .background(Color.blue)
        .clipShape(Circle())

      Text(text)
        .font(.system(size: 14))
        .foregroundColor(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
  }
}
