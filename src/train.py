from pathlib import Path
from tqdm import tqdm
from pprint import pprint
import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
from model import CaptchaModel


def train(model, data_loader, optimizer):
    model.train()                           # put model to train mode
    final_loss = 0
    tk = tqdm(data_loader, total=len(data_loader))

    for data in tk:
        for k, v in data.items():
            # loop through the 2 elements in the dict (images:tensor, targets:tensor)
            # and put it on the gpu
            data[k] = v.to(config.device)

        optimizer.zero_grad()

        # Pass **data because data is a dict. model takes 2 params - images, targets
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        final_loss += loss.item()

    return final_loss / len(data_loader)


def eval(model, data_loader):
    model.eval()                            # Eval mode
    final_loss = 0
    final_preds = []

    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))

        for data in tk:
            for k, v in data.items():
                data[k] = v.to(config.device)

            batch_preds, loss = model(**data)
            final_loss += loss.item()
            final_preds.append(batch_preds)

        return final_preds, final_loss / len(data_loader)


def decode_predictions(preds, encoder):
    """
    :param preds: Predictions from the model
    :param encoder: Label Encoder
    """
    preds = preds.permute(1, 0, 2)          # [bs, ts, preds]
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()


    captcha_preds = []
    batch_size = preds.shape[0]
    for img_index in range(batch_size):
        temp_preds = []                     # store preds of each image
        timesteps = preds[img_index]        # (75, ) pred labels for each timestep
        for t in timesteps:
            t = t - 1                       # subtract 1 from the labels
            if t == -1:
                temp_preds.append("*")      # indicates an unknown char
            else:
                pred_label = encoder.inverse_transform([t])[0]
                temp_preds.append(pred_label)

        # Now we have all 75 preds for a single image
        # Concat all 75 chars into a string and append them
        captcha_preds.append("".join(temp_preds))

    return captcha_preds


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
        train_loss = train(model, train_loader, optimizer)
        valid_preds, valid_loss = eval(model, test_loader)

        # Print out the actual label and predicted labels
        # Loop through and pass each batch to the decode function
        valid_preds_tmp  = []
        for vp in valid_preds:
            vp = decode_predictions(vp, le)
            valid_preds_tmp.extend(vp)
        valid_preds = valid_preds_tmp
        pprint(list(zip(test_orig_targets, valid_preds))[:5])

        print(f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {valid_loss}")


if __name__ == "__main__":
    run_training()
