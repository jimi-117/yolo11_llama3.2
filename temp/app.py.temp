import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import json
from datetime import datetime

@st.cache_resource
def load_model():
    """Load and cache the YOLO model"""
    try:
        model_path = "yolo11x.pt"
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def process_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Preprocess the input image"""
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size)
    return image

def save_feedback(feedback_data: dict):
    """Save feedback to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feedback_file = Path("feedback") / f"feedback_{timestamp}.json"
    feedback_file.parent.mkdir(exist_ok=True)
    
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)

def main():
    # Page config
    st.set_page_config(
        page_title="YOLO Object Detection Demo",
        page_icon="🔍",
        layout="wide"
    )

    # Application UI
    st.title("YOLO Object Detection Demo")
    st.write("Upload an image to perform object detection")

    # Detection settings in sidebar
    confidence = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

    # Load model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Load and preprocess image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = process_image(img)
            
            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)
            
            # Display input image in left column
            with col1:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Input Image", width=500)

            if model is not None:
                with st.spinner("Running object detection..."):
                    # Run inference
                    start_time = time.time()
                    results = model(img, conf=confidence, verbose=False)
                    inference_time = time.time() - start_time

                    # Display results
                    for result in results:
                        # Draw detection results in right column
                        annotated_img = result.plot()
                        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                        with col2:
                            st.image(annotated_img, caption="Detection Results", width=500)

                        # Display detection info
                        boxes = result.boxes
                        if len(boxes) > 0:
                            st.success(f"Detection completed! Processing time: {inference_time:.2f} seconds")
                            st.write(f"Objects detected: {len(boxes)}")
                            
                            # Store feedback data
                            feedback_data = {
                                "timestamp": datetime.now().isoformat(),
                                "image_name": uploaded_file.name,
                                "inference_time": inference_time,
                                "detections": []
                            }

                            # Display detailed results with feedback
                            for i, box in enumerate(boxes):
                                conf = box.conf[0].item()
                                cls = result.names[box.cls[0].item()]
                                
                                # Create columns for detection info and feedback
                                info_col1, info_col2 = st.columns([3, 1])
                                
                                with info_col1:
                                    st.write(f"Detection #{i+1}: {cls} (confidence: {conf:.2%})")
                                
                                with info_col2:
                                    feedback_key = f"feedback_{i}"
                                    feedback = st.radio(
                                        "Is this detection accurate?",
                                        options=["Undecided", "Accurate", "Inaccurate"],
                                        key=feedback_key,
                                        horizontal=True
                                    )

                                    # Store feedback in data structure
                                    detection_data = {
                                        "id": i,
                                        "class": cls,
                                        "confidence": float(conf),
                                        "feedback": feedback,
                                        "bbox": box.xyxy[0].tolist()
                                    }
                                    feedback_data["detections"].append(detection_data)
                                
                                # Add separator between detections
                                st.markdown("---")

                            # Save feedback button
                            if st.button("Save Feedback"):
                                save_feedback(feedback_data)
                                st.success("Feedback saved successfully!")
                        else:
                            st.info("No objects detected")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("👆 Please upload an image file")

if __name__ == "__main__":
    main()