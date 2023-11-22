import os
from pathlib import Path
import json
import numpy as np
import cv2
from rich.progress import track
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection

script_dir = os.path.dirname(os.path.realpath(__file__))

def read_keypoints(json_file):
    data = json.load(open(json_file, "r"))
    data = list(data.values())
    
    filenames = []
    keypoints = []
    
    for record in data:        
        im_keypoints = [] 
        [im_keypoints.extend((x, y)) for (x, y) in record["face_landmarks"]]
        
        keypoints.append(im_keypoints)
        filenames.append(record["file_name"])
        
    result = { f:np.array(k) for f, k in zip(filenames, keypoints) }
    return result

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

def detect_face(face_det, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_det.process(img)
    dets = results.detections or []
    if len(dets) == 0:
        return None

    bboxes = [get_bbox(det, img.shape) for det in dets]
    bbox = max(bboxes, key=lambda x: x[2] * x[3])
    return bbox

def shift_keypoints(keypoints, bbox):
    bx, by, bw, bh = bbox
    norm_keypoints = []

    for x, y in keypoints.reshape(-1, 2):
        x = max(min(x - bx, bw), 0)
        y = max(min(y - by, bh), 0)

        norm_keypoints.extend((x, y))

    return norm_keypoints

def main(ims_path, kp_file):
    face_det = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    im_files = [x for x in os.listdir(ims_path) if x.endswith(".png")]
    keypoints = read_keypoints(kp_file)

    for im_file in track(im_files):
        im_keypoints = keypoints[im_file]
        im = cv2.imread(os.path.join(ims_path, im_file))

        bbox = detect_face(face_det, im)
        if bbox is None:
            continue

        im_keypoints = shift_keypoints(im_keypoints, bbox)
        obj = {"bbox": bbox, "bbox_keypoints": im_keypoints}

        basename = Path(im_file).stem
        json.dump(obj, open(os.path.join(ims_path, basename + ".json"), "w"))


if __name__ == "__main__":
    main(f"{script_dir}/images/", f"{script_dir}/all_data.json")