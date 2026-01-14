import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Page config
st.set_page_config(page_title="AI Object Detection", page_icon="🔍", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 AI Object Detection Tool")
st.markdown("Detect objects in real-time using cutting-edge AI technology!")

# Load model
@st.cache_resource
def load_model():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    return processor, model

with st.spinner("🤖 Loading AI model..."):
    processor, model = load_model()

st.success("✅ Model ready to detect!")

# Detection function for live video
def detect_objects_cv2(frame, threshold=0.85):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Run detection
    inputs = processor(images=pil_image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Convert outputs
    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    
    # Draw on frame
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
        box = [int(i) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        confidence = score.item()
        
        color = colors[idx % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
        
        # Create label
        text = f"{label_name}: {confidence:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (box[0], box[1] - text_height - 10), 
                     (box[0] + text_width, box[1]), color, -1)
        
        # Draw text
        cv2.putText(frame, text, (box[0], box[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# Video transformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.threshold = 0.85
        self.frame_count = 0
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process every 3rd frame for performance
        self.frame_count += 1
        if self.frame_count % 3 == 0:
            img = detect_objects_cv2(img, self.threshold)
        
        return img

# Detection function with better labeling for static images
def detect_objects(image, threshold=0.85):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Convert outputs to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    
    # Try to load a nice font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    detections = []
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    
    for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 3)
        
        color = colors[idx % len(colors)]
        
        # Draw rectangle
        draw.rectangle(box, outline=color, width=4)
        
        # Create label with background
        text = f"{label_name} ({confidence:.2f})"
        
        # Get text bounding box
        bbox = draw.textbbox((box[0], box[1] - 25), text, font=font)
        
        # Draw background rectangle for text
        draw.rectangle(bbox, fill=color)
        
        # Draw text
        draw.text((box[0], box[1] - 25), text, fill="white", font=font)
        
        detections.append({
            "label": label_name,
            "confidence": confidence,
            "box": box
        })
    
    return image, detections

# Sidebar
st.sidebar.header("⚙️ Settings")
detection_threshold = st.sidebar.slider("Detection Confidence", 0.5, 1.0, 0.85, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Statistics")
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
st.sidebar.metric("Total Objects Detected", st.session_state.total_detections)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Features")
st.sidebar.markdown("✅ **Live video detection**")
st.sidebar.markdown("✅ Upload images")
st.sidebar.markdown("✅ Detect from URLs")
st.sidebar.markdown("✅ Color-coded bounding boxes")
st.sidebar.markdown("✅ Real-time labeling")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Video", "📤 Upload Image", "🔗 URL", "ℹ️ About"])

with tab1:
    st.subheader("🎥 Real-Time Video Object Detection")
    st.markdown("**Live video feed with object detection - just like YOLO!**")
    
    st.info("🎬 Click 'START' to begin live detection. Objects will be detected in real-time!")
    
    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Create video transformer instance
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.threshold = detection_threshold
            self.frame_count = 0
            
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Process every 2nd frame for better performance
            self.frame_count += 1
            if self.frame_count % 2 == 0:
                img = detect_objects_cv2(img, self.threshold)
            
            return img
    
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔆 **Good Lighting**\nEnsure proper lighting")
    with col2:
        st.markdown("📏 **Distance**\nKeep objects in frame")
    with col3:
        st.markdown("⚙️ **Threshold**\nAdjust in sidebar")

with tab2:
    st.subheader("📤 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="📁 Original Image", use_container_width=True)
        
        if st.button("🔍 Detect Objects", key="upload_detect"):
            with st.spinner("🔎 Analyzing image..."):
                result_image, detections = detect_objects(image.copy(), detection_threshold)
                st.session_state.total_detections += len(detections)
            
            with col2:
                st.image(result_image, caption=f"✨ Detected {len(detections)} objects", use_container_width=True)
            
            if detections:
                st.success(f"🎯 Found {len(detections)} objects!")
                
                # Group by object type
                object_counts = {}
                for det in detections:
                    object_counts[det['label']] = object_counts.get(det['label'], 0) + 1
                
                st.markdown("### 📊 Detection Summary")
                for obj, count in object_counts.items():
                    st.markdown(f"- **{obj}**: {count}")
                
                st.markdown("### 🔍 Detailed Results")
                for i, det in enumerate(detections, 1):
                    with st.expander(f"Object {i}: {det['label']} ({det['confidence']:.1%})"):
                        st.json(det)
            else:
                st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")

with tab3:
    st.subheader("🔗 Detect from URL")
    image_url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
    
    if image_url:
        try:
            with st.spinner("📥 Downloading image..."):
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="🌐 Image from URL", use_container_width=True)
            
            if st.button("🔍 Detect Objects", key="url_detect"):
                with st.spinner("🔎 Analyzing image..."):
                    result_image, detections = detect_objects(image.copy(), detection_threshold)
                    st.session_state.total_detections += len(detections)
                
                with col2:
                    st.image(result_image, caption=f"✨ Detected {len(detections)} objects", use_container_width=True)
                
                if detections:
                    st.success(f"🎯 Found {len(detections)} objects!")
                    
                    # Group by object type
                    object_counts = {}
                    for det in detections:
                        object_counts[det['label']] = object_counts.get(det['label'], 0) + 1
                    
                    st.markdown("### 📊 Detection Summary")
                    for obj, count in object_counts.items():
                        st.markdown(f"- **{obj}**: {count}")
                    
                    st.markdown("### 🔍 Detailed Results")
                    for i, det in enumerate(detections, 1):
                        with st.expander(f"Object {i}: {det['label']} ({det['confidence']:.1%})"):
                            st.json(det)
                else:
                    st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.info("💡 Make sure the URL is a direct link to an image file.")

with tab4:
    st.subheader("ℹ️ About This App")
    st.markdown("""
    ### 🤖 AI-Powered Object Detection
    
    This application uses state-of-the-art deep learning to detect objects in **real-time video streams** and images.
    
    **🎯 Model:** Facebook DETR (DEtection TRansformer)
    - Pre-trained on COCO dataset
    - Can detect 91 different object categories
    - Uses transformer architecture for accurate detection
    - **Real-time video processing like YOLO!**
    
    **✨ Features:**
    - 🎥 **Real-time live video detection**
    - 📤 Upload your own images
    - 🔗 Analyze images from URLs
    - 🎨 Color-coded bounding boxes
    - 🏷️ Object labels with confidence scores
    - 📊 Detection statistics
    
    **🎨 Object Categories Include:**
    Person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, 
    fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, 
    elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, and many more!
    
    **🔧 How to Use:**
    1. Go to the **Live Video** tab
    2. Click **START** to begin detection
    3. Allow camera access when prompted
    4. Watch objects being detected in real-time!
    
    **⚙️ Adjust Settings:**
    Use the sidebar to adjust the detection confidence threshold. Lower values detect more 
    objects but may include false positives. Higher values are more conservative.
    
    ---
    
    **Made with:** Streamlit + Hugging Face Transformers + PyTorch + WebRTC
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🚀 Powered by <b>Facebook DETR</b> | Built with ❤️ using Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
