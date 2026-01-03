import * as FileSystem from 'expo-file-system';
import { API_URL } from '../config/constants';

export interface FrameData {
  frame: string; // base64 encoded image
  timestamp: number;
}

class VideoStreamService {
  private isStreaming = false;
  private intervalId: NodeJS.Timeout | null = null;
  private frameCallback: ((frame: FrameData) => void) | null = null;

  /**
   * Start streaming frames to the API
   * @param onFrame Callback for each frame extracted
   */
  async startStreaming(onFrame?: (frame: FrameData) => void): Promise<void> {
    if (this.isStreaming) {
      console.warn('Already streaming');
      return;
    }

    this.isStreaming = true;
    this.frameCallback = onFrame || null;

    console.log('Started video streaming (1 frame per 2 seconds)');
  }

  /**
   * Capture a frame from Meta Wearables and send to API
   */
  async captureAndSendFrame(photoUri: string): Promise<void> {
    try {
      const timestamp = Date.now();

      // Read the image file as base64
      const base64 = await FileSystem.readAsStringAsync(photoUri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      const frameData: FrameData = {
        frame: base64,
        timestamp,
      };

      // Call the callback if provided
      if (this.frameCallback) {
        this.frameCallback(frameData);
      }

      // Send to API
      await this.sendFrameToAPI(frameData);
    } catch (error) {
      console.error('Error capturing and sending frame:', error);
      throw error;
    }
  }

  /**
   * Send frame to API for processing
   */
  private async sendFrameToAPI(frameData: FrameData): Promise<void> {
    try {
      const response = await fetch(`${API_URL}/process-frame`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: frameData.frame,
          timestamp: frameData.timestamp,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const result = await response.json();
      console.log('API response:', result);

      return result;
    } catch (error) {
      console.error('Error sending frame to API:', error);
      throw error;
    }
  }

  /**
   * Stop streaming
   */
  stopStreaming(): void {
    if (!this.isStreaming) {
      return;
    }

    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    this.isStreaming = false;
    this.frameCallback = null;

    console.log('Stopped video streaming');
  }

  /**
   * Check if currently streaming
   */
  isCurrentlyStreaming(): boolean {
    return this.isStreaming;
  }
}

export const videoStreamService = new VideoStreamService();
