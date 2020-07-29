from tqdm import tqdm
import torch
import config


def train(model, data_loader, optimizer):
    model.train()
    final_loss = 0
    tk = tqdm(data_loader, total=len(data_loader))

    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        final_loss += loss.item()

    return final_loss / len(data_loader)


def eval(model, data_loader, optimizer):
    model.eval()
    final_loss = 0
    final_preds = []
    tk = tqdm(data_loader, total=len(data_loader))

    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)
        batch_preds, loss = model(**data)
        loss.backward()
        final_loss += loss.item()
        final_preds.append(batch_preds)

    return final_preds, final_loss / len(data_loader)