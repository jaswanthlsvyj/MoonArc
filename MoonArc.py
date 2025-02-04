import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import io
from PIL import Image
from datetime import datetime
import pandas as pd

# Custom preprocessing function for Lambda layer compatibility
def effnet_preprocess(img):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(img)

# Load the trained model with compatibility settings
@st.cache_resource()
def load_model():
    tf.keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(
        'MoonArcModel.keras',
        compile=False,
        custom_objects={'effnet_preprocess': effnet_preprocess}
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def apply_clahe(image):
    """Apply CLAHE enhancement to moon images"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((clahe_l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def preprocess_image(image_bytes):
    """Full preprocessing pipeline including detection, cropping and CLAHE"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None, None, False

    # Moon detection pipeline
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moon_detected = False
    processed_image = image.copy()
    annotated_image = image.copy()

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw detection overlay
        cv2.circle(annotated_image, center, radius, (0, 255, 0), 2)
        
        # Calculate crop bounds
        y_min = max(center[1] - radius, 0)
        y_max = min(center[1] + radius, image.shape[0])
        x_min = max(center[0] - radius, 0)
        x_max = min(center[0] + radius, image.shape[1])
        
        cropped = image[y_min:y_max, x_min:x_max]
        if cropped.size != 0:
            # Apply CLAHE enhancement
            processed_image = apply_clahe(cropped)
            moon_detected = True

    return processed_image, annotated_image, moon_detected

def predict_moon_phase(image_bytes, model):
    """Classify the moon phase from processed image"""
    processed_image, annotated_image, moon_detected = preprocess_image(image_bytes)
    if processed_image is None:
        return None, None, False, "Error processing image"

    # Resize and prepare for model
    resized = cv2.resize(processed_image, (224, 224))
    img_array = tf.keras.utils.img_to_array(resized)
    img_array = tf.expand_dims(img_array, 0)
    
    # Make prediction
    predictions = model.predict(img_array)
    class_names = ['first quarter', 'full moon', 'new moon', 'no moon',
                   'third quarter', 'waning crescent', 'waning gibbous',
                   'waxing crescent', 'waxing gibbous']
    
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(tf.nn.softmax(predictions[0])) * 100
    
    return predicted_class, annotated_image, moon_detected, confidence

def main():
    st.title("🌖 Moon Phase Classifier")
    model = load_model()

    # Input method selection
    input_method = st.radio("Select Input Method:", ("Upload Image", "Camera Capture"))
    image_bytes = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose a moon image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file:
            image_bytes = uploaded_file.read()

    elif input_method == "Camera Capture":
        camera_img = st.camera_input("Take a moon photo")
        if camera_img:
            image_bytes = camera_img.read()

    if image_bytes:
        # Display original image
        original_image = Image.open(io.BytesIO(image_bytes))
        
        # Process and predict
        predicted_class, annotated_image, moon_detected, confidence = predict_moon_phase(image_bytes, model)
        
        # Convert OpenCV images to PIL format
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        # Display results
        if predicted_class == 'no moon':
            st.image(original_image, caption="Input Image", use_column_width=True)
            st.warning("No moon detected in the image")
            st.error(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(annotated_pil, caption="Processed Image with Detection", use_column_width=True)
            
            st.success(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")

            # Feedback system
            st.subheader("Provide Feedback")
            feedback = st.radio("Is this prediction correct?", ("Yes", "No"))
            
            if feedback == "No":
                correct_class = st.selectbox("Select correct moon phase:",
                    ['first quarter', 'full moon', 'new moon', 'third quarter',
                     'waning crescent', 'waning gibbous', 'waxing crescent', 'waxing gibbous'])
                
                if st.button("Submit Feedback"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_name = f"feedback_{timestamp}.jpg"
                    img_path = os.path.join("user_feedback", img_name)
                    
                    os.makedirs("user_feedback", exist_ok=True)
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Save feedback to CSV
                    feedback_data = {
                        "timestamp": [timestamp],
                        "image_path": [img_path],
                        "predicted_class": [predicted_class],
                        "correct_class": [correct_class],
                        "confidence": [confidence]
                    }
                    pd.DataFrame(feedback_data).to_csv("feedback.csv", mode='a', header=not os.path.exists("feedback.csv"))
                    st.success("Thank you for improving our model!")

if __name__ == "__main__":
    main()