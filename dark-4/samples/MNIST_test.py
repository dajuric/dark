import pickle
import numpy as np
import cv2
import glob, os
import matplotlib.pyplot as plt
from MNIST_train import MyNNBlock, MyNN #for deserialization

def get_random_sample():
    images = glob.glob("samples/db-FashionMNIST/test/**/*.png")
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
    net = pickle.load(open("samples/model.pickle", "rb"))
    im = get_random_sample()
    print(classify_sample(net, im))
    show_sample(im)


if __name__ == "__main__":
    np.seterr(over='raise')
    main()