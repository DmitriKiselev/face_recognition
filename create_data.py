#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Template doc
"""
import argparse
import cv2


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_path',
                        help='Path to save data.')

    return parser.parse_args()


def data_creator(save_path):

    # To capture video from webcam.
    cap = cv2.VideoCapture(0)
    # To use a video file as input
    # cap = cv2.VideoCapture('filename.mp4')
    count = 0
    while True:
        # Read the frame
        _, img = cap.read()
        cv2.imwrite(save_path + str(count) + '.jpg', img)
        count += 1

        # Display
        cv2.imshow('img', img)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Release the VideoCapture object
    cap.release()


def main():
    args = get_args()
    data_creator(args.save_path)

if __name__ == '__main__':
    main()
