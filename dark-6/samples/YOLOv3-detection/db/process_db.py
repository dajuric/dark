import os
import cv2
from rich.progress import track

script_dir = os.path.dirname(os.path.realpath(__file__))

def read_file(txt_file):
    f = open(txt_file, "r")
    data = dict()
    
    while True:
        im_file = f.readline().strip('\n')
        if len(im_file) == 0:
            break
        
        n_boxes = int(f.readline().strip('\n'))
        if n_boxes == 0:
            f.readline() #read empty bbox
            continue
        
        boxes = []
        for i in range(n_boxes):
            record = f.readline().strip('\n').split(" ")[:4]
            box    = [int(x) for x in record]
            boxes.append(box)
            
        data[im_file] = boxes    
        
    f.close()
    return data
        
def write_data(im_boxes, im_base_folder):
    for im_file, boxes in track(im_boxes.items()):
        imH, imW, _ = cv2.imread(os.path.join(im_base_folder, im_file)).shape
        
        im_folder = os.path.join(im_base_folder, os.path.dirname(im_file))
        bb_file   = os.path.basename(im_file).replace(".jpg", ".txt")
             
        #convert to yolo boxes
        yolo_boxes = []
        for box in boxes:
            x, y, w, h = box
            if w == 0 or h == 0:
                continue
            
            x, y, w, h = x / imW, y / imH, w / imW, h / imH  
            xC = x + w * 0.5
            yC = y + h * 0.5
            
            yolo_boxes.append([xC, yC, w, h])

        
        #write to file
        if len(yolo_boxes) == 0:
            continue
        
        f_bb = open(os.path.join(im_folder, bb_file), "w+")
        
        for box in yolo_boxes:
            row = "0 " + " ".join([str(x) for x in [xC, yC, w, h]])
            f_bb.write(row + "\n")
        
        f_bb.close() 
        
        
def main(txt_file, im_base_folder):
    im_boxes = read_file(txt_file)
    write_data(im_boxes, im_base_folder)


if __name__ == "__main__":
    main(f"{script_dir}/train.txt", f"{script_dir}/train/")
    main(f"{script_dir}/val.txt",   f"{script_dir}/val/")