import Foundation

/// Configuration mode for LiveKit
enum LiveKitMode: String, Codable {
    case api        // Fetch token from API
    case manual     // Manual token entry
}

/// Configuration for LiveKit connection and publishing
struct LiveKitConfig: Codable {
    let mode: LiveKitMode
    let apiURL: String?          // API base URL (e.g., http://192.168.1.100:8000)
    let serverURL: String?       // LiveKit server URL (manual mode only)
    let token: String?           // Manual token (manual mode only)
    let roomName: String?        // Manual room name (manual mode only)
    let agentName: String?       // Agent name for backend processing
    let enableVideo: Bool
    // Audio is always enabled - use mute button to control during streaming

    // MARK: - Initializers

    /// Create API mode configuration
    static func apiMode(apiURL: String, agentName: String? = nil, enableVideo: Bool = true) -> LiveKitConfig {
        return LiveKitConfig(
            mode: .api,
            apiURL: apiURL,
            serverURL: nil,
            token: nil,
            roomName: nil,
            agentName: agentName,
            enableVideo: enableVideo
        )
    }

    /// Create manual mode configuration
    static func manualMode(serverURL: String, token: String, roomName: String, agentName: String? = nil, enableVideo: Bool = true) -> LiveKitConfig {
        return LiveKitConfig(
            mode: .manual,
            apiURL: nil,
            serverURL: serverURL,
            token: token,
            roomName: roomName,
            agentName: agentName,
            enableVideo: enableVideo
        )
    }

    // MARK: - Info.plist Configuration

    /// Load default configuration from Info.plist (set via Config.xcconfig)
    static func loadFromInfoPlist() -> LiveKitConfig? {
        guard let liveKitConfig = Bundle.main.object(forInfoDictionaryKey: "LiveKitConfig") as? [String: String],
              let apiConfig = Bundle.main.object(forInfoDictionaryKey: "APIConfig") as? [String: String] else {
            return nil
        }

        let serverURL = liveKitConfig["ServerURL"]
        let devToken = liveKitConfig["DevToken"]
        let devRoomName = liveKitConfig["DevRoomName"]
        let agentName = liveKitConfig["AgentName"]
        let apiBaseURL = apiConfig["BaseURL"]
        // Note: APIKey and APISecret are available but not used in current implementation
        // They would be needed for generating tokens on-device

        // Helper to check if string is truly empty (handles "" from xcconfig)
        func isNonEmpty(_ value: String?) -> Bool {
            guard let value = value, !value.isEmpty else { return false }
            // Treat literal "" as empty (happens when xcconfig has = "" instead of just =)
            return value != "\"\""
        }

        // If dev token is provided, use manual mode (for development)
        if let devToken = devToken, isNonEmpty(devToken),
           let serverURL = serverURL, isNonEmpty(serverURL),
           let roomName = devRoomName, isNonEmpty(roomName) {
            return LiveKitConfig.manualMode(
                serverURL: serverURL,
                token: devToken,
                roomName: roomName,
                agentName: agentName
            )
        }

        // Otherwise, use API mode if API URL is available (for production)
        if let apiBaseURL = apiBaseURL, isNonEmpty(apiBaseURL) {
            return LiveKitConfig.apiMode(apiURL: apiBaseURL, agentName: agentName)
        }

        // No default config available
        // User will need to configure via settings
        return nil
    }

    // MARK: - UserDefaults Persistence

    private static let configKey = "LiveKitConfig"

    /// Load configuration from UserDefaults, fallback to Info.plist defaults
    static func loadFromUserDefaults() -> LiveKitConfig? {
        guard let data = UserDefaults.standard.data(forKey: configKey) else {
            // No saved config, try loading from Info.plist
            return loadFromInfoPlist()
        }

        let decoder = JSONDecoder()
        return try? decoder.decode(LiveKitConfig.self, from: data)
    }

    /// Save configuration to UserDefaults
    func saveToUserDefaults() {
        let encoder = JSONEncoder()
        if let data = try? encoder.encode(self) {
            UserDefaults.standard.set(data, forKey: Self.configKey)
        }
    }

    /// Remove configuration from UserDefaults
    static func clearFromUserDefaults() {
        UserDefaults.standard.removeObject(forKey: configKey)
    }

    // MARK: - Validation

    /// Validate that the configuration has required fields based on mode
    var isValid: Bool {
        switch mode {
        case .api:
            return apiURL != nil && !apiURL!.isEmpty
        case .manual:
            guard let serverURL = serverURL, let token = token, let roomName = roomName else {
                return false
            }
            return !serverURL.isEmpty && !token.isEmpty && !roomName.isEmpty
        }
    }

    /// Validate that URLs are properly formatted
    var hasValidURLs: Bool {
        switch mode {
        case .api:
            guard let apiURL = apiURL, let url = URL(string: apiURL) else { return false }
            return url.scheme == "http" || url.scheme == "https"
        case .manual:
            guard let serverURL = serverURL, let url = URL(string: serverURL) else { return false }
            return url.scheme == "ws" || url.scheme == "wss"
        }
    }
}

/// Session credentials from API
struct LiveKitSessionCredentials {
    let sessionId: String?  // Optional: only set when using API mode (UUID as string)
    let serverURL: String
    let token: String
    let roomName: String
}
