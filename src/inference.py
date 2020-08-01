import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch

import numpy as np
from PIL import Image
from pathlib import Path
from sklearn import preprocessing

import config
from model import CaptchaModel
from train import decode_predictions


def preproc_image(image_path):
    """
    :param image_path: path to the test image
    :return: {'images': image tensor}
    """
    image = Image.open(image_path).convert("RGB")

    plt.imshow(image)
    plt.show()

    transformer = transforms.Compose([
        transforms.Resize((config.image_height, config.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = torch.as_tensor(transformer(image), dtype=torch.float)
    image = torch.unsqueeze(image, dim=0)

    return {'images': image}


def remove_blanks(preds):
    pred_str = preds[0]
    new_str = ""

    prev_char = ""
    for c in pred_str:
        if c == "*":
            prev_char = ""
            continue
        elif c == prev_char:
            continue
        else:
            new_str += c

    return list(new_str)


def get_predictions(image_path, model_path):
    classes = ['2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']
    le = preprocessing.LabelEncoder()
    le.fit(sorted(classes))
    n_classes = len(classes)

    model = CaptchaModel(num_chars=n_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data = preproc_image(image_path)

    with torch.no_grad():
        preds, _ = model(**data)

    # Now decode the preds
    preds = decode_predictions(preds, le)
    preds = remove_blanks(preds)
    print(preds)


if __name__ == "__main__":
    image_path = Path(config.output_dir)/"test2.png"
    model_path = Path(config.output_dir)/"captcha_model.pkl"

    get_predictions(image_path, model_path)