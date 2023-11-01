from train import *

device = "cuda" if torch.cuda.is_available() else "cpu"

augs = A.Compose([
    A.Resize(140, 140),
    A.Normalize()
], 
keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))

test_df  = pd.read_csv("data/test_frames_keypoints.csv")
testset = FacialKeyDataset(test_df,  "data/test/",  augs)
imIndex = 23
im, keys = testset[imIndex]

model = torch.load("model/FacialModel.pth", map_location=device)
model.eval()
with torch.no_grad():
    im = im.unsqueeze(0).to(device)
    predictedKeys = model(im)
    showItems(im.squeeze(), keys, predictedKeys)
