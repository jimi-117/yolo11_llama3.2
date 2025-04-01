import streamlit as st
import requests
import cv2
import numpy as np
from datetime import datetime
import json

# base API URL
API_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="YOLO Object Detection Demo",
        page_icon="🔍",
        layout="wide"
    )

    st.title("YOLO Object Detection Demo")
    st.write("Upload an image to perform object detection")

    # detection settings
    confidence = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # send request to Yolo API
            files = {"file": uploaded_file}
            params = {"confidence": confidence}
            response = requests.post(f"{API_URL}/detect/", files=files, params=params)
            result = response.json()

            # display input and output images horizontally
            col1, col2 = st.columns(2)
            
            with col1:
                img_bytes = uploaded_file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Input Image", width=400)

            with col2:
                annotated_img = np.array(result["annotated_image"])
                st.image(annotated_img, caption="Detection Results", width=400)

            # show detection results
            if result["detections"]:
                st.success(f"Detection completed! Processing time: {result['inference_time']:.2f} seconds")
                st.write(f"Objects detected: {len(result['detections'])}")

                feedback_data = {
                    "timestamp": datetime.now().isoformat(),
                    "image_name": uploaded_file.name,
                    "inference_time": result["inference_time"],
                    "detections": []
                }

                for i, detection in enumerate(result["detections"]):
                    info_col1, info_col2 = st.columns([3, 1])
                    
                    with info_col1:
                        st.write(f"Detection #{i+1}: {detection['class_name']} "
                               f"(confidence: {detection['confidence']:.2%})")
                    
                    with info_col2:
                        feedback = st.radio(
                            "Is this detection accurate?",
                            options=["Undecided", "Accurate", "Inaccurate"],
                            key=f"feedback_{i}",
                            horizontal=True
                        )

                        detection["feedback"] = feedback
                        feedback_data["detections"].append(detection)

                    st.markdown("---")

                if st.button("Save Feedback"):
                    response = requests.post(f"{API_URL}/feedback/", json=feedback_data)
                    if response.status_code == 200:
                        st.success("Feedback saved successfully!")
                    else:
                        st.error("Failed to save feedback")
            else:
                st.info("No objects detected")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("👆 Please upload an image file")

if __name__ == "__main__":
    main()