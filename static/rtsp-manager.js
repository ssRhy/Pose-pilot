/**
 * RTSP Stream Manager for Pose Pilot
 * Handles real-time RTSP communication with the backend
 */

class RtspManager {
  constructor() {
    this.isRunning = false;
    this.pollInterval = null;
    this.pollingRate = 200; // ms
    this.listeners = {
      onFrame: [],
      onStatus: [],
      onError: [],
      onAdvice: [],
    };
    this.lastStatus = null;
    this.lastAdvice = null;
    this.badPostureStartTime = null;
    this.badPostureDuration = 0;
    this.lastErrorWasNoResults = false; // Track if the last error was a "no results" error
    this.connectionAttempts = 0; // Track connection attempts

    // Stats
    this.stats = {
      framesProcessed: 0,
      badPostureDetections: 0,
      goodPostureDetections: 0,
      adviceGenerated: 0,
    };
  }

  /**
   * Start the RTSP stream
   * @returns {Promise} A promise that resolves when the stream is started
   */
  async start() {
    try {
      // Reset error tracking when starting a new connection
      this.lastErrorWasNoResults = false;
      this.connectionAttempts = 0;
      
      const response = await fetch("/rtsp/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}) // Add empty JSON object as body
      });

      const data = await response.json();

      if (data.success) {
        this.isRunning = true;
        this._notifyListeners("onStatus", { running: true });
        this._startPolling();
        return { success: true };
      } else {
        this._notifyListeners("onError", {
          message: "无法启动RTSP流",
          details: data.message || "未知错误",
        });
        return { success: false, error: data };
      }
    } catch (error) {
      this._notifyListeners("onError", {
        message: "启动RTSP流时出错",
        details: error.message || "网络连接错误",
      });
      return { success: false, error };
    }
  }

  /**
   * Stop the RTSP stream
   * @returns {Promise} A promise that resolves when the stream is stopped
   */
  async stop() {
    try {
      const response = await fetch("/rtsp/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const data = await response.json();

      if (data.success) {
        this._stopPolling();
        this.isRunning = false;
        this._notifyListeners("onStatus", { running: false });
        return { success: true };
      } else {
        this._notifyListeners("onError", {
          message: "Failed to stop RTSP stream",
          details: data,
        });
        return { success: false, error: data };
      }
    } catch (error) {
      this._notifyListeners("onError", {
        message: "Error stopping RTSP stream",
        details: error,
      });
      return { success: false, error };
    }
  }

  /**
   * Check if the RTSP stream is running
   * @returns {Promise} A promise that resolves with the status
   */
  async checkStatus() {
    try {
      const response = await fetch("/rtsp/status");
      const data = await response.json();

      this.isRunning = data.running;
      this._notifyListeners("onStatus", { running: this.isRunning });

      return { success: true, running: this.isRunning };
    } catch (error) {
      this._notifyListeners("onError", {
        message: "Error checking RTSP status",
        details: error,
      });
      return { success: false, error };
    }
  }

  /**
   * Capture a baseline posture
   * @returns {Promise} A promise that resolves with the baseline data
   */
  async captureBaseline() {
    try {
      const response = await fetch("/capture_baseline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const data = await response.json();

      if (data.success) {
        return { success: true, data };
      } else {
        this._notifyListeners("onError", {
          message: "Failed to capture baseline",
          details: data,
        });
        return { success: false, error: data };
      }
    } catch (error) {
      this._notifyListeners("onError", {
        message: "Error capturing baseline",
        details: error,
      });
      return { success: false, error };
    }
  }

  /**
   * Generate a posture report
   * @returns {Promise} A promise that resolves with the report data
   */
  async generateReport() {
    try {
      const response = await fetch("/rtsp/report");
      const data = await response.json();

      if (data.success && data.result) {
        if (data.result.advice) {
          this.lastAdvice = data.result.advice;
          this._notifyListeners("onAdvice", { advice: data.result.advice });
          this.stats.adviceGenerated++;
        }
        return { success: true, data: data.result };
      } else {
        this._notifyListeners("onError", {
          message: "Failed to generate report",
          details: data,
        });
        return { success: false, error: data };
      }
    } catch (error) {
      this._notifyListeners("onError", {
        message: "Error generating report",
        details: error,
      });
      return { success: false, error };
    }
  }

  /**
   * Get the latest stats
   * @returns {Object} Current stats
   */
  getStats() {
    return { ...this.stats };
  }

  /**
   * Add event listener
   * @param {string} event - Event name: 'frame', 'status', 'error', 'advice'
   * @param {Function} callback - Callback function
   */
  addEventListener(event, callback) {
    const eventName = `on${event.charAt(0).toUpperCase() + event.slice(1)}`;
    if (this.listeners[eventName]) {
      this.listeners[eventName].push(callback);
    }
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function to remove
   */
  removeEventListener(event, callback) {
    const eventName = `on${event.charAt(0).toUpperCase() + event.slice(1)}`;
    if (this.listeners[eventName]) {
      this.listeners[eventName] = this.listeners[eventName].filter(
        (cb) => cb !== callback
      );
    }
  }

  /**
   * Private: Start polling for frames
   */
  _startPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
    }

    // Get the first frame immediately
    this._fetchLatestFrame();

    // Set up polling at the specified rate
    this.pollInterval = setInterval(
      () => this._fetchLatestFrame(),
      this.pollingRate
    );
  }

  /**
   * Private: Stop polling for frames
   */
  _stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
  }

  /**
   * Private: Fetch the latest frame from the server
   */
  async _fetchLatestFrame() {
    if (!this.isRunning) return;

    try {
      const response = await fetch("/rtsp/latest");
      const data = await response.json();

      // Check if there was an error response from the server
      if (!data.success) {
        console.error("Server reported an error:", data.message || "Unknown error");
        
        // Special handling for "No RTSP detection results available" error
        if (data.message === "No RTSP detection results available") {
          // This is a common startup condition, handle it more gracefully
          // Only notify if this is not a repeated error (to avoid spamming the UI)
          if (!this.lastErrorWasNoResults) {
            this._notifyListeners("onError", {
              message: "等待RTSP数据",
              details: "正在等待摄像头数据，请确保摄像头已连接并正常工作"
            });
            this.lastErrorWasNoResults = true;
          }
        } else {
          // For other errors, show the normal error message
          this._notifyListeners("onError", {
            message: "RTSP连接错误",
            details: data.message || "未知错误"
          });
          this.lastErrorWasNoResults = false;
        }
        return;
      }
      
      // Reset the error tracking flag when we get a successful response
      this.lastErrorWasNoResults = false;

      // Check if we have a result
      if (!data.result) {
        console.error("No result data in response");
        return;
      }

      // Check if there's an error in the result
      if (data.result.error) {
        console.error("RTSP error:", data.result.error);
        this._notifyListeners("onError", {
          message: "RTSP流错误",
          details: data.result.message || data.result.error
        });
        // Still update the frame if there's an image
        if (data.result.image) {
          this._notifyListeners("onFrame", data.result);
        }
        return;
      }

      // Process successful result
      this.stats.framesProcessed++;

      // Track posture status changes
      const currentStatus = data.result.posture_status;

      if (currentStatus === "bad") {
        this.stats.badPostureDetections++;

        // Track bad posture duration
        if (this.lastStatus !== "bad") {
          this.badPostureStartTime = Date.now();
        } else if (this.badPostureStartTime) {
          this.badPostureDuration = Math.floor(
            (Date.now() - this.badPostureStartTime) / 1000
          );

          // If bad posture for more than 30 seconds and no recent advice, generate a report
          if (
            this.badPostureDuration > 30 &&
            (!this.lastAdvice || Date.now() - this.lastAdviceTime > 60000)
          ) {
            this.generateReport();
            this.lastAdviceTime = Date.now();
          }
        }
      } else if (currentStatus === "good") {
        this.stats.goodPostureDetections++;

        // Reset bad posture tracking
        this.badPostureStartTime = null;
        this.badPostureDuration = 0;
      } else if (currentStatus === "error") {
        // Handle error status specifically
        this._notifyListeners("onError", {
          message: "RTSP连接错误",
          details: data.result.message || "未知错误"
        });
      }

      this.lastStatus = currentStatus;

      // Add duration and stats to the result
      const enrichedResult = {
        ...data.result,
        badPostureDuration: this.badPostureDuration,
        stats: { ...this.stats },
      };

      // Notify listeners
      this._notifyListeners("onFrame", enrichedResult);

      // Update advice if available
      if (data.result.advice) {
        this.lastAdvice = data.result.advice;
        this.lastAdviceTime = Date.now();
        this._notifyListeners("onAdvice", { advice: data.result.advice });
        this.stats.adviceGenerated++;
      }
    } catch (error) {
      console.error("Error fetching latest frame:", error);
      // Notify of connection errors to help debugging
      this._notifyListeners("onError", {
        message: "无法连接到服务器",
        details: error.message
      });
    }
  }

  /**
   * Private: Notify all listeners of an event
   * @param {string} eventName - Event name
   * @param {Object} data - Event data
   */
  _notifyListeners(eventName, data) {
    if (this.listeners[eventName]) {
      this.listeners[eventName].forEach((callback) => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${eventName} listener:`, error);
        }
      });
    }
  }
}

// Create a singleton instance
const rtspManager = new RtspManager();
