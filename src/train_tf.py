import os
import argparse
import tensorflow as tf
from src.model_tf import build_tf_model

def load_data(data_dir='data', image_size=(224, 224), batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'training'),
        image_size=image_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'testing'),
        image_size=image_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    # Normalizing images to [-1, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, class_names

def train_tf_model(name='khondwani', epochs=10, batch_size=32):
    train_ds, val_ds, class_names = load_data(batch_size=batch_size)
    model = build_tf_model(num_classes=len(class_names))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{name}_model.tensorflow'
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='khondwani')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    train_tf_model(name=args.name, epochs=args.epochs, batch_size=args.batch_size)
