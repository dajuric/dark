import dark
import dark.tensor as dt
from model import Resnet18
import numpy as np

import os
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

print(f"Running on: {'cuda' if dt.is_cuda() else 'cpu'}")
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{script_dir}/model.pth"

def get_bbox(det, im_shape, enlarge_bbox = 0.1):
    imH, imW, _ = im_shape

    normalized_bb = det.location_data.relative_bounding_box
    x = int(min(max(normalized_bb.xmin   * imW, 0), imW))
    y = int(min(max(normalized_bb.ymin   * imH, 0), imH)) 
    w = int(min(max(normalized_bb.width  * imW + x, 0) - x, imW)) 
    h = int(min(max(normalized_bb.height * imH + y, 0) - y, imH))

    #enlarge bbox
    x -= int(w * enlarge_bbox); w += int(w * enlarge_bbox) * 2
    y -= int(h * enlarge_bbox); h += int(h * enlarge_bbox) * 2
    
    #clip box
    x = max(min(x, imW), 0); w = min(w, x + imW)
    y = max(min(y, imH), 0); h = min(h, y + imH)
     
    return x, y, w, h

def draw_keypoints(im, keypoints, bbox, color = (0, 255, 0)):
    bx, by, bw, bh = bbox

    for x, y in keypoints.reshape(-1, 2):
        x = int(x * bw) + bx
        y = int(y * bh) + by

        cv2.circle(im, (x, y), 3, color, -1)

def eval_network(net, patch):
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = cv2.resize(patch, (128, 128))
    patch = (patch - 127) / 127
    
    patch = np.moveaxis(patch, 2, 0)
    patch = np.expand_dims(patch, 0)

    keypoints = net(patch)
    keypoints = keypoints.data.squeeze()

    return keypoints.get()

def detect_face(face_det, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_det.process(img)
    dets = results.detections or []
    if len(dets) == 0:
        return None

    bboxes = [get_bbox(det, img.shape) for det in dets]
    bbox = max(bboxes, key=lambda x: x[2] * x[3])
    return bbox

def main():
    net = dark.load(model_path)
    net.eval()
    
    face_det = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, image = cap.read()
        if not ret: break

        bbox = detect_face(face_det, image)
        if bbox is not None:
            x, y, w, h = bbox
            patch = image[y:y+h, x:x+w, :]
            
            keypoints = eval_network(net, patch)
            draw_keypoints(image, keypoints, (x, y, w, h))

        cv2.imshow("Keypoint detection", image)
        if cv2.waitKey(5) == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()