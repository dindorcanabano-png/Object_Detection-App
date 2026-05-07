import streamlit as st
import av
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- TITLE ----------------
st.title("🎥 Live Object Detection & Tracing")

# ---------------- FACE MODEL ----------------
face_net = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- CALLBACK ----------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    results = model.track(img, persist=True, conf=0.5, verbose=False)
    annotated = results[0].plot()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_net.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ---------------- STREAM ----------------
webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)
