#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Template doc
"""
import argparse
import os


from random import shuffle
from shutil import copy


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_size', type=float,
                        help='Size of new train set (0..1)')
    parser.add_argument('--source_dir', type=str,
                        help='Dir of sets')
    parser.add_argument('--train_set_dir', type=str,
                        help='Dir for new training set')
    parser.add_argument('--val_set_dir', type=str,
                        help='Dir for new val set')

    return parser.parse_args()


def split(set_dir, train_set_dir, val_set_dir, train_size):
    """

    Parameters
    ----------
    set_dir :
        Direction of set that should be splitted
    train_set_dir:
        Direction to save created train set
    val_set_dir:
        Direction to save created validation set
    train_size: float
        Size of new train set

    Returns
    -------

    """
    person_path = str(set_dir) + 'Dmitry/'
    unknown_path = str(set_dir) + 'unknown_person/'
    person_list = os.listdir(person_path)
    unknown_list = os.listdir(unknown_path)
    shuffle(person_list), shuffle(unknown_list)

    person_train_set = person_list[:int(len(os.listdir(str(set_dir) + 'Dmitry/'
                                                       )) * train_size)]
    person_val_list = person_list[int(len(os.listdir(str(set_dir) + 'Dmitry/'))
                                      * train_size):]
    unknown_train_set = unknown_list[:int(len(os.listdir(str(set_dir) +
                                          'unknown_person/')) * train_size)]
    unknown_val_set = unknown_list[int(len(os.listdir(str(set_dir) +
                                           'unknown_person/')) * train_size):]

    for file in person_train_set:
        copy(person_path + file, train_set_dir + 'Dmitry/')

    for file in person_val_list:
        copy(person_path + file, val_set_dir + 'Dmitry/')

    for file in unknown_train_set:
        copy(unknown_path + file, train_set_dir + 'unknown_person/')

    for file in unknown_val_set:
        copy(unknown_path + file, val_set_dir + 'unknown_person/')
    return


def main():
    """Application entry point."""
    args = get_args()
    split(args.source_dir, args.train_set_dir, args.val_set_dir,
          args.train_size)


if __name__ == '__main__':
    main()
