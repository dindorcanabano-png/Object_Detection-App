import streamlit as st
import cv2
import av

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="Object Detection and Tracing",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 Live Object Detection & Tracing")
st.write("YOLO + Face Detection")

# ---------------- LOAD YOLO MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- FACE DETECTION MODEL ----------------
@st.cache_resource
def load_face_model():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_net = load_face_model()

# ---------------- VIDEO CALLBACK ----------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # YOLO detection
    results = model(img, conf=0.5, verbose=False)
    annotated = results[0].plot()

    # Face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_net.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ---------------- WEBCAM STREAM ----------------
webrtc_streamer(
    key="ai-stream",
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

st.caption("Runs locally only. No video is uploaded.")