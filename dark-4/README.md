# Module 4

We came a long way since the first module. To be able to write our first full fledged sample - Fashion MNIST classification, with training and validation loop, we just need some utilities: *Dataset*, *DataLoader* and transformation classes to be able to read and pre-process images from a disk. 

## Dataset class

Dataset class in PyTorch is a class responsible for loading samples and their corresponding labels - one by one, not in batches.
There are only two methods which are required to be implemented:
1) __len__ - so we can know the number of samples, and we call this function via standard *length* operator
2) __getitem__ - so we can use built-in indexing method to access the data, just like in a standard list

```python
class Dataset():
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()
```

PyTorch implements a concrete class derived from *Dataset* - *ImageFolder*. This class is designed for image datasets which are divided into subfolders, where each subfolder is treated as a label. E.g. in MNIST dataset we would have folders *0*, *1*, *2*, ... and those would be the labels as well. Let us implement this class as well.

```python
class ImageFolder(Dataset):
    def __init__(self, root_folder, imgT = None, lblT = None):
        super().__init__()
        self.root_folder = root_folder
        self.imgT = imgT
        self.lblT = lblT

        self.img_names = self._get_im_paths(self.root_folder)
        self.labels = self._get_labels(self.img_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img = self._load_image(self.img_names[index])
        lbl = self.labels[index]

        if self.imgT:
            img = self.imgT(img)
        if self.lblT:
            lbl = self.lblT(lbl)

        return img, lbl

    def _get_im_paths(self, root_folder):
        #get jpg, png, bmp, webp images - sorted in natural order
        ...

        return im_paths

    def _get_labels(self, imgNames):
        #get distinct labels from folder names by preserving their order
        ...

        return labels

    def _load_image(self, im_path):
        #get image in a gray, RGB, RGBA or unchanged format
        ...

        return image
```

In the implementation above, an implementation of some utility functions is omitted for the sake of clarity and simplicity. 
First, we get all image paths using *_get_im_paths*, then we convert all sub-folders into integer labels using *_get_labels* and finally we implement *getitem* and *len* methods. *imgT* and *lblT* are optional transform functions that operate on an image or a label. The rest is self-explanatory.

## DataLoader class

*DataLoader* class takes dataset and carries batch loading for us. It has two methods:
1) __len__ - which returns the number of batches
2) __iter__ - which returns a batch iterator object, thus enabling us to use the same dataloader multiple times

```python
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
```

The interesting part is an internal class *_DataLoaderIterator* which does the actual work of loading data. It receives a dataloader object and exposes a single method *next* which loads a batch of data. 

Each time when the *next* method is called via for loop for example, the next batch is loaded. Very important property of a dataloader is an ability to shuffle samples so the training procedure could be executed properly - otherwise the validation accuracy would suffer greatly. 
Usually variable *batch* in the *next* function of the implementation below would have two dimensions (image, label). However, we might return multiple objects, that is why the implementation has the second loop *for dim...* which supports such case.
The end result is a tuple, where each dimension of a tuple is a NumPy array for each sample element. 

```python
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
            raise StopIteration #if we loaded everything, we are done

        # calculate start and stop index
        batch = [] #list of NumPy arrays (e.g. usually images array and labels array)
        start_idx = self.sample_index
        stop_idx  = min(self.sample_index + self.batch_size, len(self.dataset))

        #process each sample in an order dictated by sample_indices
        for idx in range(start_idx, stop_idx):
            sIdx = self.sample_indices[idx]
            sample = self.dataset[sIdx]
            self.sample_index += 1

            #for each dimension of a sample:
            for dim in range(len(sample)):
               sampleItem = np.expand_dims(sample[dim], 0) #...expand the batch dimension

               #if we need to expand our batch dimension do it (e.g. case of multiple images and a label)
               if len(batch) <= dim: 
                   batch.append(sampleItem)     
               else: #otherwise concat sample to a corresponding batch dimension
                   batch[dim] = np.concatenate((batch[dim], sampleItem), axis=0)

        return tuple(batch) #return list to tuple
```

## Transforms

In order to pre-process our dataset and to regularize our model we need image transformation functions / objects. Those functions are passed to a dataset which calls them after an image is loaded (take a look at the above *ImageFolder* implementation).

It is common to compose such transform objects into a sequence like so:

```python
transforms = Compose(
        Resize(28, 28),
        Grayscale(),
        FlipHorizontal(),
        Normalize(0.5, 0.5),
        ToTensorV2()
    )
```

An example of transformation is shown bellow. We can see that a transformation implements a single method *call* which enables a Python object to be called as a function (obj(params)).

```python
class FlipHorizontal():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im):
        if random.random() < self.p:
            return im

        flippedIm = cv2.flip(im, 1)
        return flippedIm
```

Very useful object is a *Compose* object that calls passed transforms in a sequence. Its entire implementation is shown below.

```python
class Compose():
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, im):
        transformedIm = im
        for imT in self.transforms:
            transformedIm = imT(transformedIm)

        return transformedIm
```

## Sample - FashionMNIST classification

We have reached a point where you can follow an official PyTorch tutorial for MNIST classification (but using feed forward NN model) to implement a full fledged sample - because the API is very similar! 
The entire implementation is provided on GitHub. However due to a larger amount of code and abundance of other PyTorch tutorials the code sample will be skipped.
Only some crucial differences will be mentioned here:
1) when dealing with variables in a graph, you have to explicitly use *.value* property to access underlying NumPy array (tensor)
2) *CrossEntropyLoss* only receives one-hot encoded inputs due to simplicity
3) The implemented *TotensorV2* corresponds to implementation of *Albumentations* [TODO] library rather than standard *ToTensor* PyTorch function. In other words input arrays are not scaled with 255


## Remarks and Final Thoughts
This tutorial sets a significant milestone. Many crucial parts are implemented which enable us to use custom feed forward NNs like in PyTorch. 
However, in order to work with images convolutional neural networks are preferred. Those layers/extensions to our library will be implemented next.

## References   


