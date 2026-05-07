import streamlit as st
import av

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# ---------------- SAFE MODEL LOADING ----------------
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- SAFE OPENCV ----------------
try:
    import cv2
except Exception:
    st.error("OpenCV failed. Fix requirements.txt (opencv-python-headless + libgl1).")
    st.stop()

# ---------------- VIDEO CALLBACK ----------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # YOLO tracking
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    annotated = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ---------------- WEBCAM STREAM ----------------
from streamlit_webrtc import webrtc_streamer, WebRtcMode

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
