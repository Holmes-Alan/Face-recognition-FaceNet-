# Face recognition

Simple implementation on face recognition using FaceNet with training and testing

By Zhi-Song Liu, Wan-Chi Siu and H. Anthony Chan

# Dependencies
    miniconda
    Python > 3.5
    OpenCV library
    Pytorch > 1.3
    Scikit-image
    NVIDIA GPU + CUDA
    ![facenet](https://github.com/timesler/facenet-pytorch)
    
# Training

1. Collect your own training facial images
```sh
$ python data_collect.py --folder_name xxx
```
"xxx" is the name of the person, it is also the name of the folder
The "xxx" folder is created under folder "train_data"

2. Train the face recognition model
```sh
$ python face_recog_train.py
```

# Testing
run the following command
```sh
$ python face_recog_test.py
```
