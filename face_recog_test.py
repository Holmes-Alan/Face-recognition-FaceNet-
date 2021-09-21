import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
from torchvision import transforms

def collate_fn(x):
    return x[0]



class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, key_to_classname):

        self.key_to_classname = key_to_classname
        self.last_box = None

    def _draw(self, frame, boxes, probs, landmarks, name):
        """
        Draw landmarks and boxes for each face detected
        """
        try:
            print('drawing')
            for box, prob, ld, id in zip(boxes, probs, landmarks, name):
                # Draw rectangle on frame

                cv2.putText(frame, id, (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        except:
            print('not draw box')
            pass

        return frame

    def recog(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        x, _ = mtcnn(face, return_prob=True)
        x = torch.stack([x]).to(device)
        out = resnet(x).detach().cpu()
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        print("Prediction : ", self.key_to_classname[topclass.cpu().numpy()[0][0]], ", Score: ", topk.cpu().numpy()[0][0])
        name = [self.key_to_classname[topclass.cpu().numpy()[0][0]][1]]

        return name

    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            try:
                name = self.recog(frame)
                boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
                if self.last_box is not None:
                    # print('last_box: ', self.last_box)
                    cx_0, cy_0 = (self.last_box[0][0] + self.last_box[0][2]) // 2, (self.last_box[0][1] + self.last_box[0][3]) // 2
                    cx_1, cy_1 = (boxes[0][0] + boxes[0][2]) // 2, (boxes[0][1] + boxes[0][3]) // 2
                    w_0, h_0 = self.last_box[0][2] - self.last_box[0][0], self.last_box[0][3] - self.last_box[0][1]
                    w_1, h_1 = boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]

                    factor_center = 0.3
                    new_cx = cx_0 + factor_center * (cx_1 - cx_0)
                    new_cy = cy_0 + factor_center * (cy_1 - cy_0)

                    factor_hw = 0.3
                    new_w = w_0 + factor_hw * (w_1 - w_0)
                    new_h = h_0 + factor_hw * (h_1 - h_0)

                    boxes = [[int(new_cx - new_w // 2), int(new_cy - new_h // 2),
                             int(new_cx + new_w // 2), int(new_cy + new_h // 2)]]

                self.last_box = boxes

                # draw on frame
                self._draw(frame, boxes, probs, landmarks, name)
                print(name)
                # draw on frame

            except:
                pass

            # Show the frame
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Run the app
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = torch.load('models/own_face.pth')
resnet = resnet.to(device).eval()

trans = transforms.Compose([
    transforms.ToTensor(),
    # fixed_image_standardization
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Load Imagenet Synsets
with open('image/id_class_train.txt', 'r') as f:
    synsets = f.readlines()
synsets = [x.strip() for x in synsets]
key_to_classname = [line.split(':') for line in synsets]

fcd = FaceDetector(key_to_classname)
fcd.run()
