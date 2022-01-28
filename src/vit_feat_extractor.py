from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import torch
import pdb
import cv2
import os

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# https://huggingface.co/docs/transformers/model_doc/vit


def np_to_pil(img):
    return Image.fromarray(img)

class ImageClassificationCollator:
    def __init__(self, feature_extractor):
      self.feature_extractor = feature_extractor

    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings

def image_feature_extraction(image):
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape) # [1, 197, 768]
    return last_hidden_states
    # pdb.set_trace()

import random
import numpy as np
from tqdm import tqdm

def main():
    path = "data/TRAINING"
    saving_path = "data/TRAINING/vit_features.npy"
    vit_feats_arr = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in tqdm(sorted(filenames)):
            img = cv2.imread(os.path.join(dirpath, filename))
            pil_img = np_to_pil(img)
            vit_feats_arr.append(image_feature_extraction(pil_img))
            val = random.random()
            if val <= 0.02:
                np.save(file=saving_path,
                        arr=np.array(vit_feats_arr),
                        allow_pickle=True
                        )

    np.save(file=saving_path,
            arr=np.array(vit_feats_arr),
            allow_pickle=True)


if __name__ == '__main__':
    main()