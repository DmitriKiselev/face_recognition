#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Template doc
"""
import argparse
import torch
import torchvision

from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path',
                        help='Path of saved model')
    parser.add_argument('--file_path',
                        help='Path of photo to predict')

    return parser.parse_args()


def predict(model_path, file_path):
    """

    Parameters
    ----------
    model_path: path
        Path of the saved model
    file_path : path
        Path to data for predicition
    Returns
    -------

    """
    data_transforms = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    show_prediction = mpimg.imread(file_path)
    img = Image.open(file_path)
    t_img = data_transforms(img).unsqueeze(axis=0)
    model = torchvision.models.resnet18(pretrained=False, num_classes=2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    out = model(t_img)
    _, preds = torch.max(out, 1)
    plt.figure()
    plt.imshow(show_prediction, cmap='gray')
    plt.suptitle('Dmitry' if preds[0].tolist() == 0 else 'Unknown')
    plt.show()
    return


def main():
    """Application entry point."""
    args = get_args()
    predict(args.model_path, args.file_path)


if __name__ == '__main__':
    main()

