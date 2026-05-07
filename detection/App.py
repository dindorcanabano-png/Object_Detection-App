import streamlit as st
from ultralytics import YOLO
import av
import cv2
from streamlit_webrtc import webrtc_streamer

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("Live Detection")

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model(img, verbose=False)
    annotated = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="test",
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
)
