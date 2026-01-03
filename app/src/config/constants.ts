// API Configuration
export const API_URL = __DEV__ ? 'http://localhost:8000' : 'https://api.blindsighted.app';

// Streaming Configuration
export const STREAMED_FPS = 0.5; // Frames per second (0.5 = 1 frame every 2 seconds)
export const STREAM_INTERVAL_MS = 1000 / STREAMED_FPS; // Convert FPS to milliseconds

// Meta Wearables Configuration
export const META_APP_ID = '856891110384461';

// Image Quality
export const STREAM_IMAGE_QUALITY = 80; // 0-100
