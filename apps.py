import streamlit as st
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms

from src.model_torch import BrainTumorCNN
from src.model_tf import build_tf_model
from src.dataset import get_class_names

# Load models and class names
class_names = get_class_names()

torch_model = BrainTumorCNN(num_classes=len(class_names))
torch_model.load_state_dict(torch.load('models/khondwani_model.torch', map_location='cpu'))
torch_model.eval()

tf_model = tf.keras.models.load_model('models/khondwani_model.tensorflow')

# Preprocessing functions
def preprocess_image_torch(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    return transform(image).unsqueeze(0)

def preprocess_image_tf(image):
    image = image.resize((224, 224))
    image = np.array(image) / 127.5 - 1.0
    return np.expand_dims(image, axis=0)

# Prediction functions
def predict_with_torch(image):
    input_tensor = preprocess_image_torch(image)
    with torch.no_grad():
        output = torch_model(input_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

def predict_with_tf(image):
    input_tensor = preprocess_image_tf(image)
    predictions = tf_model.predict(input_tensor)
    return class_names[np.argmax(predictions)]

# Streamlit UI
st.title("Brain Tumor Classifier")

st.sidebar.header("Model Selection")
framework = st.sidebar.radio("Choose a Framework", ('TensorFlow', 'PyTorch'))

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated here

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            prediction = predict_with_torch(image) if framework == "PyTorch" else predict_with_tf(image)
        st.success(f"Predicted Tumor Type: **{prediction}**")

# CSS Styling
def local_css(file_name):
    full_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(full_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom CSS
local_css("static/css/styles.css")
