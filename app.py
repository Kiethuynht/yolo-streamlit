import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
from ultralytics import YOLO

# Load YOLO model
@st.cache_resource()
def load_model():
    model = YOLO("best.pt")  # Thay bằng đường dẫn mô hình của bạn nếu có
    return model

model = load_model()

# Streamlit UI
st.title("🔍 YOLO Object Detection App")
st.write("Upload an image and let YOLO detect objects!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    # Convert file to PIL Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Detecting objects...")

    # Convert image to array and process with YOLO
    results = model(image)

    # Draw bounding boxes on image
    for result in results:
        img = np.array(image)
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            label = result.names[int(result.boxes.cls[i])]
            conf = box[5] if len(box) > 5 else 0.0
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Convert image back to PIL format for display
    st.image(img, caption="Detected Objects", use_container_width=True)
    
# Webcam Option
use_camera = st.checkbox("Use Webcam")

if use_camera:
    cap = cv2.VideoCapture(0)
    st.write("Starting webcam...")
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        # Draw bounding boxes
        for result in results:
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box[:4])
                label = result.names[int(result.boxes.cls[i])]
                conf = box[5] if len(box) > 5 else 0.0
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame
        frame_placeholder.image(frame, channels="BGR")

    cap.release()
