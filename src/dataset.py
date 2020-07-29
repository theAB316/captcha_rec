import torchvision.transforms as transforms
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

# dont throw error in case of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None):
        """
        :param image_paths: list of paths
        :param targets: list of list of chars. for e.g. ['a','2','3','b','c']
        :param resize: tuple - (height, width)
        """

        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.transformer = transforms.Compose([
            # Resize takes (rows, cols)
            transforms.Resize((self.resize[0], self.resize[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # Read and convert 4 channel to 3 channel image
        # PIL size [image.size] gives (width, height)
        image = Image.open(self.image_paths[index]).convert("RGB")
        targets = self.targets[index]

        image = self.transformer(image)
        return {
            "images": torch.as_tensor(image, dtype=torch.float),
            "targets": torch.as_tensor(targets, dtype=torch.long),
        }