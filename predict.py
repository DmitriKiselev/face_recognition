#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Template doc
"""
import argparse
import torch
import torchvision
import csv

from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path',
                        help='Path of saved model')
    parser.add_argument('--data_dir',
                        help='Direction of data to predict')
    parser.add_argument('--csv_path',
                        help='Path for saving csv')
    return parser.parse_args()


def csv_writer(data, path):
    """

    Parameters
    ----------
    data:
        List with data for writing in the csv
    path:
        Path for saving created csv
    Returns
    -------

    """
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File', 'Class'])
        for i in range(len(data)):
            writer.writerow(data[i].split(','))


def predict(model_path, data_dir):
    """

    Parameters
    ----------
    model_path: path
        Direction to the saved model
    data_dir:
        Direction to data for predicition
    Returns
    -------

    """
    data_transforms = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    pred_dataset = ImageFolder(data_dir, data_transforms)
    pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet18(pretrained=False, num_classes=2)
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    predictions = []
    for inputs, labels in pred_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.tolist()
        for i in range(len(preds)):
            predictions.append(preds[i])
    for i in range(len(predictions)):
        predictions[i] = str(Path(pred_dataset.samples[i][0]).name) + ',' + (
            'Dmitry' if predictions[i] == 0 else 'unknown_person')
    return predictions


def main():
    """Application entry point."""
    args = get_args()
    pred = predict(args.model_path, args.data_dir)
    csv_writer(pred, args.csv_path)


if __name__ == '__main__':
    main()

