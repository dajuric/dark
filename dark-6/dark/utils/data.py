import os, glob
import math
import numpy as np
import cv2

class Dataset():
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()


class ImageFolder(Dataset):
    def __init__(self, rootFolder, imgT = None, lblT = None):
        super().__init__()
        self.rootFolder = rootFolder
        self.imgT = imgT
        self.lblT = lblT

        self.imgNames = sorted(glob.glob(os.path.join(rootFolder, "**", "*.png")))
        self.labels = self.get_labels(self.imgNames)

    def __len__(self):
        return len(self.imgNames)

    def __getitem__(self, index):
        img = cv2.imread(self.imgNames[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = self.labels[index]

        if self.imgT:
            img = self.imgT(img)
        if self.lblT:
            lbl = self.lblT(lbl)

        return img, lbl

    def class_count(self):
        return len(set(self.labels))

    @staticmethod
    def get_labels(imgNames):
        labels = []
        distinctLabels = []  #distinct labels by the order of arrival

        for imgName in imgNames:
            imFolder = os.path.basename(os.path.dirname(imgName))
            if imFolder not in distinctLabels:
                distinctLabels.append(imFolder) 

            labels.append(distinctLabels.index(imFolder))   

        return labels



class DataLoader():
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        batch_count = math.ceil(len(self.dataset) / self.batch_size)
        return batch_count

    def __iter__(self):
        return _DataLoaderIterator(self)

class _DataLoaderIterator():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        self.batch_size = self.dataloader.batch_size
        self.sample_index = 0

        self.sample_indices = np.array(np.arange(0, len(self.dataset)))
        if dataloader.shuffle: 
            self.sample_indices = np.random.permutation(self.sample_indices)

    def __next__(self):
        if self.sample_index >= len(self.dataset):
            raise StopIteration

        batch = []
        start_idx = self.sample_index
        stop_idx  = min(self.sample_index + self.batch_size, len(self.dataset))

        for idx in range(start_idx, stop_idx):
            sIdx = self.sample_indices[idx]
            sample = self.dataset[sIdx]
            self.sample_index += 1

            for dim in range(len(sample)):
               sampleItem = np.expand_dims(sample[dim], 0)

               if len(batch) <= dim: 
                   batch.append(sampleItem)     
               else:
                   batch[dim] = np.concatenate((batch[dim], sampleItem), axis=0)

        return tuple(batch)
