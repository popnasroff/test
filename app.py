import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
import cv2

# Page config
st.set_page_config(page_title="AI Object Detection", page_icon="🔍", layout="wide")

# Title
st.title("🔍 AI Object Detection Tool")
st.markdown("Detect objects in images using AI - upload, paste URL, or use your webcam!")

# Load model
@st.cache_resource
def load_model():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    return processor, model

with st.spinner("Loading AI model..."):
    processor, model = load_model()

st.success("✅ Model loaded successfully!")

# Detection function
def detect_objects(image, threshold=0.9):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Convert outputs to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 3)
        
        # Draw rectangle
        draw.rectangle(box, outline="red", width=3)
        
        # Draw label
        text = f"{label_name}: {confidence}"
        draw.text((box[0], box[1] - 10), text, fill="red")
        
        detections.append({
            "label": label_name,
            "confidence": confidence,
            "box": box
        })
    
    return image, detections

# Sidebar
st.sidebar.header("⚙️ Settings")
detection_threshold = st.sidebar.slider("Detection Threshold", 0.5, 1.0, 0.9, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**How to use:**")
st.sidebar.markdown("1. Choose input method")
st.sidebar.markdown("2. Adjust threshold if needed")
st.sidebar.markdown("3. Click 'Detect Objects'")

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🔗 URL", "📸 Webcam"])

with tab1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("🔍 Detect Objects", key="upload_detect"):
            with st.spinner("Detecting objects..."):
                result_image, detections = detect_objects(image.copy(), detection_threshold)
            
            with col2:
                st.image(result_image, caption=f"Detected {len(detections)} objects", use_container_width=True)
            
            if detections:
                st.success(f"Found {len(detections)} objects!")
                st.json(detections)
            else:
                st.warning("No objects detected. Try lowering the threshold.")

with tab2:
    st.subheader("Detect from URL")
    image_url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
    
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            if st.button("🔍 Detect Objects", key="url_detect"):
                with st.spinner("Detecting objects..."):
                    result_image, detections = detect_objects(image.copy(), detection_threshold)
                
                with col2:
                    st.image(result_image, caption=f"Detected {len(detections)} objects", use_container_width=True)
                
                if detections:
                    st.success(f"Found {len(detections)} objects!")
                    st.json(detections)
                else:
                    st.warning("No objects detected. Try lowering the threshold.")
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

with tab3:
    st.subheader("Use Webcam")
    st.markdown("📸 Capture a photo from your webcam")
    
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Captured Image", use_container_width=True)
        
        if st.button("🔍 Detect Objects", key="webcam_detect"):
            with st.spinner("Detecting objects..."):
                result_image, detections = detect_objects(image.copy(), detection_threshold)
            
            with col2:
                st.image(result_image, caption=f"Detected {len(detections)} objects", use_container_width=True)
            
            if detections:
                st.success(f"Found {len(detections)} objects!")
                st.json(detections)
            else:
                st.warning("No objects detected. Try lowering the threshold.")

# Footer
st.markdown("---")
st.markdown("**Model:** Facebook DETR (Detection Transformer) | **Framework:** Hugging Face Transformers")
