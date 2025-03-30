import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import pickle
from tqdm import tqdm


def load_model():
    base_model = Xception(weights='imagenet', include_top=False, pooling='avg')
    return Model(inputs=base_model.input, outputs=base_model.output)


def extract_features(image_folder, model):
    features = {}
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in tqdm(image_files, desc="[INFO] Extracting features"):
        img_path = os.path.join(image_folder, img_name)

        image = load_img(img_path, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape((1, 299, 299, 3))
        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)
        image_id = os.path.splitext(img_name)[0]
        features[image_id] = feature

    return features


def save_features(features, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    image_folder = "../Dataset/Images"
    output_file = "../outputs/image_features.pkl"

    print("[INFO] Loading Xception model...")
    model = load_model()

    print(f"[INFO] Extracting features from: {image_folder}")
    features = extract_features(image_folder, model)

    print(f"[INFO] Saving features to: {output_file}")
    save_features(features, output_file)

    print(f"[INFO] Done! Total images processed: {len(features)}")
