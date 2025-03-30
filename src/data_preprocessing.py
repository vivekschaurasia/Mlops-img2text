import os
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle


def load_captions(captions_file):
    with open(captions_file, 'r') as file:
        lines = file.readlines()[1:] 

    mapping = {}
    for line in lines:
        tokens = line.strip().split(',')
        if len(tokens) < 2:
            continue

        image_id = tokens[0].split('.')[0]
        caption = " ".join(tokens[1:]).lower()
        caption = re.sub(r'[^a-z\s]', '', caption)
        caption = re.sub(r'\s+', ' ', caption).strip()

        caption = f'startseq {caption} endseq'

        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)

    return mapping


def create_tokenizer(captions):
    all_captions = [cap for cap_list in captions.values() for cap in cap_list]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer


def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)


def get_max_length(captions):
    all_captions = [cap for cap_list in captions.values() for cap in cap_list]
    return max(len(caption.split()) for caption in all_captions)


if __name__ == "__main__":
    # Example usage
    #captions_path = "C:\Users\vivek\OneDrive\Desktop\MLOps image2text\Dataset\captions.txt"
    captions_path = "C:/Users/vivek/OneDrive/Desktop/MLOps image2text/Dataset/captions.txt"
    save_path = "../outputs/tokenizer.pkl"

    print("[INFO] Loading and cleaning captions...")
    caption_map = load_captions(captions_path)

    print("[INFO] Creating tokenizer...")
    tokenizer = create_tokenizer(caption_map)
    save_tokenizer(tokenizer, save_path)

    vocab_size = len(tokenizer.word_index) + 1
    max_length = get_max_length(caption_map)

    print(f"[INFO] Vocabulary Size: {vocab_size}")
    print(f"[INFO] Max Caption Length: {max_length}")

    
