import { useRef, useState } from "react";
import "./App.css";

function App() {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  const [cameraOn, setCameraOn] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordedVideo, setRecordedVideo] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setCameraOn(true);
    } catch (error) {
      alert("Please allow camera access.");
      console.error(error);
    }
  };

  const startRecording = () => {
    if (!streamRef.current) return;

    chunksRef.current = [];
    setPrediction(null);

    const recorder = new MediaRecorder(streamRef.current);
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      const videoURL = URL.createObjectURL(blob);
      setRecordedVideo(videoURL);
    };

    recorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  const deleteRecording = () => {
    if (recordedVideo) {
      URL.revokeObjectURL(recordedVideo);
    }
    setRecordedVideo(null);
    setPrediction(null);
  };

  const recordAgain = () => {
    deleteRecording();
    if (streamRef.current) {
      startRecording();
    }
  };

  const predictVideo = async () => {
    if (!recordedVideo) return;

    setLoading(true);

    try {
      const response = await fetch(recordedVideo);
      const blob = await response.blob();

      const formData = new FormData();
      formData.append("file", blob, "recording.webm");

      const result = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!result.ok) {
        const errorBody = await result.text();
        throw new Error(errorBody || "Prediction failed");
      }

      const data = await result.json();
      setPrediction(data);
    } catch (error) {
      console.error(error);
      alert("Could not connect to the backend or run the model.");
    } finally {
      setLoading(false);
    }
  };

  const stopCamera = () => {
    if (recording) {
      stopRecording();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOn(false);
  };

  const recognizedSign = prediction?.sign || "";
  const confidence = prediction
    ? `${(prediction.confidence * 100).toFixed(1)}%`
    : "";
  const sentence = prediction?.sentence || "";
  const translation = prediction?.translation || sentence || "";

  return (
    <div className="app">
      <nav className="navbar">
        <div className="logo">MSLI</div>

        <div className="nav-links">
          <button>Home</button>
          <button className="active">Recognition</button>
        </div>

        <select className="language-select">
          <option>English</option>
          <option>Telugu</option>
          <option>Hindi</option>
        </select>
      </nav>

      <main className="main-container">
        <h1>Sign Language Recognition</h1>

        <p className="subtitle">
          Convert your sign language gestures into text
        </p>

        <div className="top-layout">
          <section className="card camera-section">
            <div className="section-header">
              <h2>Live Camera</h2>

              <span className={recording ? "status recording" : "status"}>
                {recording ? "Recording" : cameraOn ? "Ready" : "Camera Off"}
              </span>
            </div>

            <div className="camera-box">
              {!cameraOn && (
                <div className="camera-message">
                  <div className="camera-icon">Camera</div>
                  <p>Camera is not started</p>
                </div>
              )}

              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className={cameraOn ? "camera-video" : "hidden"}
              />
            </div>

            <div className="controls">
              {!cameraOn && (
                <button className="btn primary" onClick={startCamera}>
                  Start Camera
                </button>
              )}

              {cameraOn && !recording && (
                <button className="btn record" onClick={startRecording}>
                  Start Recording
                </button>
              )}

              {recording && (
                <button className="btn stop" onClick={stopRecording}>
                  Stop Recording
                </button>
              )}

              {cameraOn && !recording && (
                <button className="btn secondary" onClick={stopCamera}>
                  Stop Camera
                </button>
              )}
            </div>
          </section>

          <section className="card result-section">
            <h2>Recognition Result</h2>

            <div className="result-block">
              <h3>Recognized Words</h3>
              <div className="result-box">
                <span className={recognizedSign ? "" : "placeholder"}>
                  {recognizedSign || "No sign recognized yet"}
                </span>
              </div>
            </div>

            <div className="result-block">
              <h3>Generated Sentence</h3>
              <div className="result-box sentence-box">
                <span className={sentence ? "" : "placeholder"}>
                  {sentence || "Generated sentence will appear here"}
                </span>
              </div>
            </div>

            <div className="result-block">
              <h3>Translation</h3>
              <div className="result-box translation-box">
                <span className={translation ? "" : "placeholder"}>
                  {translation || "Translation will appear here"}
                </span>
              </div>
            </div>

            {prediction && (
              <div className="result-block">
                <h3>Confidence</h3>
                <div className="result-box">
                  <span>{confidence}</span>
                </div>
              </div>
            )}

            <button className="audio-btn" disabled>
              Play Audio
            </button>
          </section>
        </div>

        <section className="card recorded-section">
          <h2>Recorded Video</h2>

          {recordedVideo ? (
            <div className="recorded-content">
              <video
                src={recordedVideo}
                controls
                className="recorded-video"
              />

              <div className="video-actions">
                <button className="btn delete" onClick={deleteRecording}>
                  Delete
                </button>

                <button className="btn secondary" onClick={recordAgain}>
                  Record Again
                </button>

                <button
                  className="btn primary"
                  onClick={predictVideo}
                  disabled={loading}
                >
                  {loading ? "Recognizing..." : "Recognize Sign"}
                </button>
              </div>
            </div>
          ) : (
            <div className="no-video">
              <p>No video recorded yet.</p>
              <small>
                Start recording and stop when you finish your gesture.
              </small>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
