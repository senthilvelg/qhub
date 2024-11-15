import os

from PIL import Image
from torch.utils.data import Dataset

'''
This class is used to load image/text annotation pair dataset.
annodations_file contains path to image and the annotation for the image in tab seperated format
img_dir contain the images. this is the root directory of the images. Images can be structured under multiple sub directories
'''


class ImageTextDataset(Dataset):
    def __init__(self, annotations_file, img_dir, processor):
        self.img_dir = img_dir
        self.processor = processor
        self.image_captions = []

        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_name, caption = line.strip().split('\t')
                self.image_captions.append((img_name, caption))

    def __len__(self):
        return len(self.image_captions)

    def __getitem__(self, idx):
        img_name, caption = self.image_captions[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        return image, caption
