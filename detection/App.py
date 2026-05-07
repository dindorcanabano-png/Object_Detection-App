import streamlit as st
import av
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="YOLO Live Detection", layout="wide")
st.title("🎥 Live Object Detection")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model(img, verbose=False)
    annotated = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="yolo",
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False}
)
