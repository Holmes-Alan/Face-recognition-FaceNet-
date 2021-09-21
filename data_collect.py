import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder_name', type=str, help='name of the image folder')
opt = parser.parse_args()

folder = 'train_data/' + opt.folder_name
try:
    os.mkdir(folder)
except OSError:
    print ("Creation of the directory %s failed" % folder)
else:
    print ("Successfully created the directory %s " % folder)
# Run the app
cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = cap.read()
    cv2.imshow('capture scene', frame)
    name = folder + '/' + str(i) + '.png'
    cv2.imwrite(name, frame)
    i = i + 1
    print(i)
    cv2.waitKey(1)
    if i > 300:
    	break

cap.release()
cv2.destroyAllWindows()