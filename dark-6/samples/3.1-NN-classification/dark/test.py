import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import dark
from model import MyNNBlock, MyNN #for deserialization
from config import *

def get_random_sample():
    images = glob.glob(f"{script_dir}/../db/test/**/*.png")
    imPath = np.random.choice(images, 1)[0]

    im = cv2.imread(imPath)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im

def classify_sample(net, im):
    im = im / 255
    im = (im - 0.5) / 0.5
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, 0)

    net.eval()
    out = net(im)
    out = np.argmax(out.value)
    return out

def show_sample(im):
    plt.figure(figsize=(1, 1))
    plt.imshow(im)
    plt.show()

def main():
    net = dark.load(model_path)
    im = get_random_sample()
    print(classify_sample(net, im))
    show_sample(im)


if __name__ == "__main__":
    main()