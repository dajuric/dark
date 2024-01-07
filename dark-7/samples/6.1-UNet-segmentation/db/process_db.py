import os
import cv2
from glob import glob
from rich.progress import track

script_dir = os.path.dirname(os.path.realpath(__file__))

def resize(img_folder):
    im_files = sorted(glob(f"{img_folder}/*.*"))

    for im_file in track(im_files, "Resizing..."):
        im = cv2.imread(im_file)
        im = cv2.resize(im, (128, 128))

        cv2.imwrite(im_file, im)

def convert(img_folder, msk_folder):
    mask_files = [x for x in os.listdir(msk_folder) if x.endswith(".gif")]

    for msk_file in track(mask_files, "Converting masks..."): 
        cap = cv2.VideoCapture(f"{msk_folder}/{msk_file}")
        _, mask = cap.read()
        cap.release()

        if mask is not None:
            cv2.imwrite(f"{img_folder}/{msk_file.replace('.gif', '.png')}", mask)


if __name__ == "__main__":
    convert(f"{script_dir}/train/", f"{script_dir}/train_masks/")
    resize(f"{script_dir}/train/") #inplace!