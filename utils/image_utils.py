from PIL import Image
import requests


def load_image(url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'):
    if 'http' in url:
        return Image.open(requests.get(url, stream=True).raw).convert('RGB')
    else:
        return Image.open(url).convert('RGB')
