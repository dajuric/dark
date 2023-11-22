## Face detection using YOLO

Yolo v3 is used as a base, where backbone is replaced with a slim BlazeFace model used in media-pipe framework. Although the model is called BlazeFace, multiple class detection is possible.

The implementation is a modified and refactored implementation from Aladin Persson with a replaced backbone.

The used database is far from ideal one, because images are cropped; however it enables fast training and the proof that the implementation works.

![Result](docs/results.png)

### Running
Go to db folder and follow the instructions to download and process database.
Go to either dark or torch folder and run 'train.py' to obtain results. They should be similar.