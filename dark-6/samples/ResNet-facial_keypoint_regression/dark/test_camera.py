import torch
from torchvision.transforms.transforms import Normalize, ToTensor
from train import FacialKeyModel #must be included so that model can be deserialized
import albumentations as A
import numpy as np

import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def getBBox(det, imageSize):
    normalizedBB = det.location_data.relative_bounding_box
    xMin = normalizedBB.xmin * imageSize[1]
    yMin = normalizedBB.ymin * imageSize[0]
    xMax = xMin + normalizedBB.width  * imageSize[1]
    yMax = yMin + normalizedBB.height * imageSize[0]

    xMin *= 0.9; xMax *= 1.1; yMin *= 0.3; yMax *= 1.1 #enlarge BB because to mimic dataset images (hack)
    return max(0, int(xMin)), max(int(yMin), 0), max(int(xMax), 0), max(int(yMax), 0)

def drawKeypoints(image, keypoints, bBox):
    xMin, yMin, xMax, yMax = bBox

    kpMin = np.array([np.min(keypoints[:, 0]), np.min(keypoints[:, 1])])
    sFac   = np.array([(xMax - xMin + 1) / 140,  (yMax - yMin + 1) / 140])

    for kpIdx in range(keypoints.shape[0]):
        kp = np.int32((keypoints[kpIdx, :] - kpMin) * sFac + kpMin * sFac + bBox[0:2])  
        cv2.circle(image, kp, 7, (255, 255, 0), -1)

    cv2.rectangle(image,  (xMin, yMin), (xMax, yMax), (0, 0, 255), 5)


def getKeypoints(net, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = A.Resize(140, 140)(image=image)["image"]
    image = A.Normalize()(image=image)["image"]
    tensor = torch.from_numpy(image)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        keypoints = net(tensor.to(device))
        keypoints = keypoints.to('cpu').squeeze().reshape(-1, 2)

    return keypoints.numpy()


def main():
    net = torch.load("model/FacialModel.pth", torch.device(device)); net.eval()
    faceDet = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    cap = cv2.VideoCapture("videoB.mp4")
    #outV = cv2.VideoWriter("o.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, (1280, 720))

    while cap.isOpened():
        ret, image = cap.read()
        if not ret: break

        #image = cv2.imread("image.png")
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = faceDet.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)       
        resultDets = results.detections or []

        for det in resultDets:
            bBox = getBBox(det, image.shape)        
            imROI = image[bBox[1]:bBox[3], bBox[0]:bBox[2]]
            keypoints = getKeypoints(net, imROI)
                
            drawKeypoints(image, keypoints, bBox)
            #mp_drawing.draw_detection(image, det)

        cv2.imshow("Face detection", image)
        if cv2.waitKey(5) == ord('q'):
            break
        #outV.write(image)

    #outV.release()

if __name__ == "__main__":
    main()