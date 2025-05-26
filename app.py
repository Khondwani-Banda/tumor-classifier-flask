import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from src.model_torch import BrainTumorCNN
from src.model_tf import build_tf_model
from src.dataset import get_class_names

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

torch_model = BrainTumorCNN(num_classes=4)
torch_model.load_state_dict(torch.load('models/khondwani_model.torch', map_location='cpu'))
torch_model.eval()
tf_model = tf.keras.models.load_model('models/khondwani_model.tensorflow')

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_torch(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

def preprocess_image_tf(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = (image / 127.5) - 1.0
    return np.expand_dims(image, axis=0)

def predict_with_torch(image_path):
    class_names = get_class_names()
    input_tensor = preprocess_image_torch(image_path)
    with torch.no_grad():
        output = torch_model(input_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

def predict_with_tf(image_path):
    class_names = get_class_names()
    input_tensor = preprocess_image_tf(image_path)
    predictions = tf_model.predict(input_tensor)
    return class_names[np.argmax(predictions)]

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files['file']
        framework = request.form['framework']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if framework == 'torch':
                prediction = predict_with_torch(filepath)
            else:
                prediction = predict_with_tf(filepath)

            image_path = '/' + filepath.replace("\\", "/")  # For Windows/Linux compatibility
            return render_template("result.html", prediction=prediction, image_path=image_path)

    return render_template("predict.html")

@app.route("/tumor-info")
def tumor_info():
    return render_template("tumor_info.html")

# IMPORTANT: for deployment

if __name__ == '__main__':
    app.run(debug=True)
