#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Template doc
"""
import argparse
import cv2
import torch
import torchvision

from torchvision import transforms
from PIL import Image


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path',
                        help='Path to model.')
    parser.add_argument('--photo_path',
                        help='Path to photo.')
    return parser.parse_args()


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

    img = Image.open(data_dir)
    t_img = data_transforms(img).unsqueeze(axis=0)
    model = torchvision.models.resnet18(pretrained=False, num_classes=2)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    out = model(t_img)
    _, preds = torch.max(out, 1)
    return preds[0]


def main():
    """Application entry point."""
    args = get_args()
    filename = args.photo_path
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # Initialize the camera (use bigger indices if you use multiple cameras)
    cap = cv2.VideoCapture(0)
    # Set the video resolution to half of the possible max resolution for better performance
    cap.set(3, 1920 / 2)
    cap.set(4, 1080 / 2)
    # Standard text that is displayed above recognized face
    text = "unknown face"
    exceptional_frames = 100
    startpoint = (0, 0)
    endpoint = (0, 0)
    color = (0, 0, 255) # Red
    while True:
        print(exceptional_frames)
        # Read frame from camera stream and convert it to greyscale
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces using cascade face detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Loop through detected faces and set new face rectangle positions
        for (x, y, w, h) in faces:
            color = (0, 0, 255)
            if not text == "unknown face":
                color = (0, 255, 0)
            startpoint = (x, y)
            endpoint = (x + w, y + h)
            face = (img[y:y + h, x:x + w])
            # Only reclassify if face was lost for at least half a second (15 Frames at 30 FPS)
            if exceptional_frames > 15:
            # Save detected face and start thread to classify it using tensorflow model
                cv2.imwrite(filename, face)
                prediction = predict(args.model_path, filename)
                if prediction.tolist() == 0:
                    text = 'Dmitry'
                else:
                    text = 'Unknown'
                exceptional_frames = 0
        # Face lost for too long, reset properties
        if exceptional_frames == 15:
            print("Exceeded exceptional frames limit")
            text = "unknown face"
            startpoint = (0, 0)
            endpoint = (1, 1)
        # Draw face rectangle and text on image frame
        cv2.rectangle(img, startpoint, endpoint, color, 2)
        textpos = (startpoint[0], startpoint[1] - 7)
        cv2.putText(img, text, textpos, 1, 1.5, color, 2)
        # Show image in cv2 window
        cv2.imshow("image", img)
        # Break if input key equals "ESC"
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        exceptional_frames += 1


if __name__ == '__main__':
    main()
