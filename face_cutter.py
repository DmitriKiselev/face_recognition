#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Template doc
"""
import argparse
import cv2
import os


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photos_path',
                        help='Path to save dir.')
    parser.add_argument('--save_path',
                        help='Path to save dir.')

    return parser.parse_args()


def face_detector(img):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_cuts = []
    for (x, y, w, h) in faces:
        # Faces are recognized with x-y (top-left point) and width-height
        face_cuts.append(img[y:y + h, x:x + w])
    # Returny images (numpy array) of detected faces
    return face_cuts


def cut_faces(photos_path, save_path):
    print("--Cutting out faces")
    i = 0
    # Loop through folders, cut out face and save face into training_images directoryr)
    for file in os.listdir(photos_path):
        image = cv2.imread(photos_path + "/" + file)
        # Detecting Faces
        faces = face_detector(image)
        for face in faces:
            # Saving the image of the face to the disk
            cv2.imwrite(save_path + "/face_{}.jpg".format(i), face)
            i += 1


def main():
    """Application entry point."""
    args = get_args()
    cut_faces(args.photos_path, args.save_path)


if __name__ == '__main__':
    main()
