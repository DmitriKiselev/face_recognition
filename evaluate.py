#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Template doc
"""

import argparse
import csv

import pandas as pd

from sklearn.metrics import recall_score, precision_score, accuracy_score
from torchvision.datasets import ImageFolder
from pathlib import Path


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir',
                        help='Direction of data to evaluate')
    parser.add_argument('--save_path',
                        help='Path for saving created csv')
    parser.add_argument('--true_path',
                        help='Path of csv with true values')
    parser.add_argument('--pred_path',
                        help='Path of csv with predicted values')
    return parser.parse_args()


def csv_maker(data_dir, path):
    """

    Parameters
    ----------
    data_dir: direction
        Direction of data to create true.csv (csv with true values)
    path:
        Path for created csv

    Returns
    -------

    """
    pred_dataset = ImageFolder(data_dir)
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File', 'Class'])
        for i in range(len(pred_dataset)):
            writer.writerow([Path(pred_dataset.samples[i][0]).name,
                            pred_dataset.samples[i][1]])


def evaluate(true_path, pred_path):
    """

    Parameters
    ----------
    true_path: path
        Path of csv with true values
    pred_path: path
        Path for csv created with predict.py

    Returns
    -------

    """
    true = pd.read_csv(true_path)
    pred = pd.read_csv(pred_path)
    y_true = true['Class'].to_list()
    y_pred = pred['Class'].to_list()
    for i in range(len(y_true)):
        if y_pred[i] == 'Dmitry':
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    print('Precision: {:4f} Recall: {:4f} Accuracy: {:4f}'.format(
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        accuracy_score(y_true, y_pred)
    ))


def main():
    """Application entry point."""
    args = get_args()
    csv_maker(args.data_dir, args.save_path)
    evaluate(args.true_path, args.pred_path)


if __name__ == '__main__':
    main()
