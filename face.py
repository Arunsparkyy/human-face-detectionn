import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from keras.models import load_model  

# Load Haar cascades for face, eyes, and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the saved model
model = load_model('face_detection_model.h5')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image, faces

def detect_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return image, eyes


def main():
    st.title("Human Faces (Object Detection)")
    st.text("Built with Streamlit and OpenCV")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        our_image = Image.open(uploaded_file)
        st.image(our_image, caption="Original Image", use_column_width=True)
        our_image = np.array(our_image)

        # Selection for enhancement type
        enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
        if enhance_type == 'Gray-Scale':
            gray_image = cv2.cvtColor(our_image, cv2.COLOR_BGR2GRAY)
            st.image(gray_image, caption="Gray-Scale Image", use_column_width=True)

        elif enhance_type == 'Contrast':
            c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
            enhancer = ImageEnhance.Contrast(Image.fromarray(our_image))
            img_output = enhancer.enhance(c_rate)
            st.image(img_output, caption="Enhanced Contrast Image", use_column_width=True)

        elif enhance_type == 'Brightness':
            c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
            enhancer = ImageEnhance.Brightness(Image.fromarray(our_image))
            img_output = enhancer.enhance(c_rate)
            st.image(img_output, caption="Enhanced Brightness Image", use_column_width=True)

        elif enhance_type == 'Blurring':
            blur_rate = st.sidebar.slider("Blur Amount", 1, 25, step=2)
            blurred_image = cv2.GaussianBlur(our_image, (blur_rate, blur_rate), 0)
            st.image(blurred_image, caption="Blurred Image", use_column_width=True)

        # Move detection type selection to sidebar
        st.sidebar.header("Detection Options")
        detection_type = st.sidebar.selectbox("Select detection type:", ["Face", "Eyes", "Smile"])
        
        if st.sidebar.button("Detect"):
            if detection_type == "Face":
                detected_image, faces = detect_faces(our_image)
                st.image(detected_image, caption='Detected Faces', use_column_width=True)
                st.write(f"Number of faces detected: {len(faces)}")
            elif detection_type == "Eyes":
                detected_image, eyes = detect_eyes(our_image)
                st.image(detected_image, caption='Detected Eyes', use_column_width=True)
                st.write(f"Number of eyes detected: {len(eyes)}")
            elif detection_type == "Smile":
                detected_image, smiles = detect_smiles(our_image)
                st.image(detected_image, caption='Detected Smiles', use_column_width=True)
                st.write(f"Number of smiles detected: {len(smiles)}")

# About Section
if st.sidebar.checkbox("About"):
    st.markdown("### Human Faces (Object Detection)", unsafe_allow_html=True)
    st.markdown("**Skills & Technologies:** Python, Data Analytics, Statistics, Visualization, Streamlit, Machine Learning, Deep Learning, Generative AI")
    st.markdown("**Domain:** Computer Vision")
    st.markdown("**Overview:** Dive into the world of human face detection! This innovative system identifies faces in images and videos with remarkable speed and accuracy, adapting seamlessly to various lighting conditions and angles.")
    st.markdown("**Technical Tags:** OpenCV, Data Preprocessing, Feature Engineering, Model Training, Model Evaluation, Hyperparameter Tuning, Deep Learning")
    st.image(r"C:\Users\nanda\Downloads\Blog-Computer-Vision-Trends-To-Adopt-In-2023.png", caption='Example of Face Detection', use_column_width=True)

    st.markdown("### About Me", unsafe_allow_html=True)
    st.markdown("Hi there! I’m Arun, a Mechanical Engineering graduate turned data enthusiast. My journey into data science has fueled my curiosity for uncovering insights and driving impactful decisions. I'm excited to explore the intersection of technology and human interaction.")
    st.markdown("[Connect with me on LinkedIn](https://www.linkedin.com/in/arun-m-j17/)")

if __name__ == '__main__':
    main()