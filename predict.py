import glob
import os
import cv2
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from generator import Generator
from discriminator import Discriminator
from PIL import Image

def to_variable(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)

def show(img): # Display rgb tensor image
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)

def show_gray(img): # Display grayscale tensor image
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)

def show_img_from_path(imgPath):
    pilImg = Image.open(imgPath)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)

def predict(model, img):
    to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
    im = to_tensor(img)
    inp = to_variable(im.unsqueeze(0), False)
    out = model(inp)
    map_out = out.cpu().data.squeeze(0)
    return map_out

def save_gray(img, path):
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    print('Image saved to ', path)
    pilImg.save(path)

pathToResizedImagesVal = '/Users/apple/Desktop/project/salicon/images256x192_val'
pathToResizedMapsVal = '/Users/apple/Desktop/project/salicon/maps256x192_val'
pathToPredictMapVal = '/Users/apple/Desktop/project/salicon/predict256x192_val'

if not os.path.exists(pathToPredictMapVal):
    os.makedirs(pathToPredictMapVal)

list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToResizedImagesVal, '*val*'))]
print(len(list_img))

 # load model

model = Generator()
pretrained_dict = torch.load('./generator.pkl')
model.load_state_dict(pretrained_dict)
if torch.cuda.is_available():
    model.cuda()
print(model)

for num in range(int(list_img)):
    imageName = list_img[num] + '.png'
    imgPath = pathToResizedImagesVal + imageName
    GroundTruthPath = pathToResizedMapsVal + imageName
    image = cv2.imread(imgPath)
    saliencyMap = predict(model, image)
    savePath = pathToPredictMapVal + str(num) + '.png'
    save_gray(saliencyMap, savePath)