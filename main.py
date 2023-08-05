import cv2
import streamlit as st
import numpy as np
import os
from video_detector import VideoReader
from face_reg import FaceRec

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def main():
    st.title("Face Detection App")
    st.write("Select a video source and see detected faces!")

    video_source = st.radio("Select Video Source", ("Webcam", "Video File"))

    face_reg = FaceRec()

    if video_source == "Webcam":
        reader = VideoReader(source="webcam", detect_faces=True, identify_faces=True)
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_file = "./temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            reader = VideoReader(source=temp_file, detect_faces=True, identify_faces=True)
        else:
            return

    stframe = st.image([])

    stop_button = st.button("Stop")

    reader.display(on_streamlit=True, stop_button=stop_button, frame_display=stframe)

    # while cap.isOpened() and not stop_button:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     processed_frame = detect_faces(frame)

    #     stframe.image(processed_frame, channels="BGR")
    
    # cap.release()

    if video_source == "Video File":
        try:
            os.remove(temp_file)
        except:
            pass

if __name__ == "__main__":
    main()
