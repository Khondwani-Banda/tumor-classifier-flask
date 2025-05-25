import argparse
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_torch_model(name):
    import torch
    from src.model_torch import BrainTumorCNN
    from src.dataset import get_dataloaders
    

    _, test_loader, class_names = get_dataloaders(batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BrainTumorCNN(num_classes=len(class_names)).to(device)
    model_path = f"models/{name}_model.torch"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== PyTorch Evaluation ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    plot_confusion_matrix(all_labels, all_preds, class_names, name + "_torch")

def evaluate_tf_model(name):
    import tensorflow as tf
    from src.train_tf import load_data


    _, test_ds, class_names = load_data(batch_size=32)
    model_path = f"models/{name}_model.tensorflow"
    model = tf.keras.models.load_model(model_path)

    y_true, y_pred = [], []

    for batch in test_ds:
        images, labels = batch
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("\n=== TensorFlow Evaluation ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    plot_confusion_matrix(y_true, y_pred, class_names, name + "_tf")

def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {title}")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/confusion_matrix_{title}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--framework', type=str, choices=['torch', 'tf'], required=True)

    args = parser.parse_args()

    if args.framework == 'torch':
        evaluate_torch_model(args.name)
    elif args.framework == 'tf':
        evaluate_tf_model(args.name)
