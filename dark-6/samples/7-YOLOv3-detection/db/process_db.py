import os, shutil
import cv2
from rich.progress import track
import mediapipe as mp
from glob import glob
mp_face_detection = mp.solutions.face_detection

script_dir = os.path.dirname(os.path.realpath(__file__))

def get_bbox(det):
    normalized_bb = det.location_data.relative_bounding_box
    x = min(max(normalized_bb.xmin, 0), 1)
    y = min(max(normalized_bb.ymin, 0), 1)
    w = min(max(normalized_bb.width  + x, 0), 1) - x
    h = min(max(normalized_bb.height + y, 0), 1) - y
     
    return x, y, w, h

def detect_faces(face_det, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_det.process(img)
    dets = results.detections or []
    if len(dets) == 0:
        return []

    bboxes = [get_bbox(det) for det in dets]
    return bboxes

def write_yolo_txt(im_boxes, im_file):
    bb_filename = im_file.replace(".jpg", ".txt")
    f_bb = open(bb_filename, "w+")

    for box in im_boxes:
        x, y, w, h = box
        
        xC = x + w * 0.5
        yC = y + h * 0.5
        yolo_box = [xC, yC, w, h]

        row = "0 " + " ".join([str(x) for x in yolo_box])
        f_bb.write(row + "\n")

    f_bb.close()


def write_detections(folder):
    face_det = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    im_files = [x for x in glob(f"{folder}/*.jpg") if x.endswith(".jpg")]

    for im_file in track(im_files, "Detecting faces..."):
        im = cv2.imread(im_file)

        bboxes = detect_faces(face_det, im)
        if len(bboxes) == 0:
            continue

        write_yolo_txt(bboxes, im_file)
       
def create_train_test_folders(test_portion = 0.1):
    print("Creating train & val")

    im_files = [x for x in glob(f"{script_dir}/**/*.jpg", recursive=True) if x.endswith(".jpg")]
    train_count = int(len(im_files) * (1 - test_portion))
    
    i = 0
    os.makedirs(f"{script_dir}/train/", exist_ok=True)
    for train_file in im_files[:train_count]:
        shutil.copy(train_file, f"{script_dir}/train/img-{i}.jpg")
        i += 1

    i = 0
    os.makedirs(f"{script_dir}/val/", exist_ok=True)
    for train_file in im_files[train_count:]:
        shutil.copy(train_file, f"{script_dir}/val/img-{i}.jpg")
        i += 1    

def cleanup():
    print("Cleaning up")

    os.remove(f"{script_dir}/readme.txt")
    os.remove(f"{script_dir}/testing.txt")
    os.remove(f"{script_dir}/training.txt")  

    shutil.rmtree(f"{script_dir}/AFLW/", ignore_errors=True)
    shutil.rmtree(f"{script_dir}/lfw_5590/", ignore_errors=True)  
    shutil.rmtree(f"{script_dir}/net_7876/", ignore_errors=True)  

if __name__ == "__main__":
    #create_train_test_folders()

    write_detections(f"{script_dir}/train/")
    write_detections(f"{script_dir}/val/")

    #cleanup()