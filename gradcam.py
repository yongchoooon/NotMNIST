from torchvision import transforms
from PIL import Image
import torch
import argparse
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2

import importlib

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
 
def model_loader(model_path):
    model = torch.load(model_path)
    model_name = model['arch']
    model = getattr(importlib.import_module('model.model'), model_name)()
    return model

def main(config):
    MODEL_PATH = config.model


    model_state_dict = torch.load(MODEL_PATH)["state_dict"]

    # load_model = ResNet152PretrainedModel()
    load_model = model_loader(MODEL_PATH)
    load_model.load_state_dict(model_state_dict)
    finalconv_name = 'layer4'

    load_model.eval()

    load_model.model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(load_model.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
    ])

    # load test image
    image_file = config.image

    img_pil = Image.open(image_file)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = load_model(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread(image_file)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('GradCAM_images/CAM.jpg', result)

if __name__ == '__main__':
    # Best model
    MODEL_PATH = './saved/models/DogBreed_ResNet152_Pretrained_Freeze_ColorJitter_lr0.005/0215_141830/model_best.pth'
    image_file = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/train/eskimo_dog/n02109961_1276.jpg'

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-m', '--model', default=MODEL_PATH, type=str,
                      help='model file path')
    args.add_argument('-im', '--image', default=image_file, type=str,
                      help='image file path')

    config = args.parse_args()
    main(config)