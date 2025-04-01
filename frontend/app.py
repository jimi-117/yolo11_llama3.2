import streamlit as st
import requests
import cv2
import numpy as np
from datetime import datetime
import json
import base64

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
            # ファイルは1度だけ読み込む
            file_bytes = uploaded_file.read()
            
            # APIリクエストに送信
            files = {"file": ("image.jpg", file_bytes, "image/jpeg")}
            params = {"confidence": confidence}
            response = requests.post(f"{API_URL}/detect/", files=files, params=params)
            result = response.json()

            # 入力画像と出力画像を横並びに表示（file_bytes を再利用）
            col1, col2 = st.columns(2)
            
            with col1:
                nparr = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Input Image", width=400)

            with col2:
                annotated_b64 = result.get("annotated_image", "")
                if annotated_b64:
                    decoded_bytes = base64.b64decode(annotated_b64)
                    if not decoded_bytes:
                        st.error("Decoded image data is empty")
                    else:
                        nparr = np.frombuffer(decoded_bytes, np.uint8)
                        annotated_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if annotated_img is None:
                            st.error("Failed to decode annotated image")
                        else:
                            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                            st.image(annotated_img_rgb, caption="Detection Results", width=400)
                else:
                    st.error("No annotated image in response")

            # 検出結果の表示とフィードバック処理
            if result.get("detections"):
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
                        st.write(f"Detection #{i+1}: {detection['class_name']} (confidence: {detection['confidence']:.2%})")
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
