from pathlib import Path
import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import engine
from model import CaptchaModel


def run_training():
    data_path = Path(config.data_dir)
    image_files = list(data_path.glob("*.png"))
    targets = []
    targets_orig = []
    targets_unique = set()

    # Loop through each file and create target list
    for file in data_path.iterdir():
        targets_orig.append(file.stem)                      # append the filename
        targets.append(list(file.stem))                     # append the list of chars
        targets_unique.update(list(file.stem))              # keep track of unique chars

    msg = "Number of target data-points: {}, \nUnique chars: {} \n"
    print(msg.format(len(targets), sorted(targets_unique)))

    # Label encode
    le = preprocessing.LabelEncoder()
    le.fit(sorted(targets_unique))
    targets_encoded = [le.transform(x) for x in targets]
    targets_encoded = np.array(targets_encoded) + 1         # adding 1 because 0 represents "unkwown"

    msg = "Encoded targets: \n{}"
    print(msg.format(targets_encoded))

    # Split dataset
    train_images, test_images, train_targets, test_targets, train_orig_targets, test_orig_targets = \
        model_selection.train_test_split(
            image_files, targets_encoded, targets_orig, test_size=0.1, random_state=42
        )

    train_dataset = dataset.ClassificationDataset(image_paths=train_images,
                                                  targets=train_targets,
                                                  resize=(config.image_height, config.image_width))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True
    )

    test_dataset = dataset.ClassificationDataset(image_paths=test_images,
                                                  targets=test_targets,
                                                  resize=(config.image_height, config.image_width))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False
    )

    model = CaptchaModel(num_chars=len(le.classes_))
    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    for epoch in range(config.epochs):
        train_loss = engine.train(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval(model, train_loader)
        print(f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {valid_loss}")




if __name__ == "__main__":
    run_training()



