import torch
import cv2
import numpy as np
import os
import argparse
from collections import OrderedDict
from PIL import Image

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ModelOutputs_resnet():
    def __init__(self, model, target_layers, target_sub_layers):
        self.model = model
        self.target_layers = target_layers
        self.target_sub_layers = target_sub_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        self.gradients = []
        for name, module in self.model.named_children(): # 모든 layer에 대해서 직접 접근
            x = module(x)
            if name == 'avgpool': # avgpool이후 fully connect하기 전 data shape을 flatten시킴
                x = torch.flatten(x,1)
                
            if name in self.target_layers: # target_layer라면 해당 layer에서의 gradient를 저장
                x.register_hook(self.save_gradient)  
                target_feature_maps = x
        return target_feature_maps, x


class GradCam_resnet:
    def __init__(self, model, target_layer_names, target_sub_layer_names):
        self.model = model
        self.model.eval()

        self.extractor = ModelOutputs_resnet(self.model, target_layer_names, target_sub_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):

        features, output = self.extractor(input)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1  # 정답이라고 생각하는 class의 index 리스트 위치의 값만 1로
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)  # numpy배열을 tensor로 변환
        one_hot = torch.sum(one_hot * output) # y^c

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy() # partial derivative of y^c with respect to A^k, (1, 2048, 7, 7) 

        target = features.cpu().data.numpy()[0, :] # A^k, (2048, 7, 7)

        weights = np.mean(grads_val, axis=(2, 3))[0, :]  # alpha_k, 논문에서의 global average pooling 식에 해당하는 부분 (1, 2048)

        grad_cam = np.zeros(target.shape[1:], dtype=np.float32)  # (7, 7)

        for i, w in enumerate(weights): # calcul grad_cam
            grad_cam += w * target[i, :, :]  # linear combination L^c_{Grad-CAM} (= A^k * alpha_k)

        grad_cam = np.maximum(grad_cam, 0)  # ReLU
        grad_cam = cv2.resize(grad_cam, (224, 224)) 
        grad_cam = grad_cam - np.min(grad_cam)  # 
        grad_cam = grad_cam / np.max(grad_cam)  # 위의 것과 해당 줄의 것은 0~1사이의 값으로 정규화하기 위한 정리
        return grad_cam


def main(config):
    MODEL_PATH = config.model
    IMAGE_PATH = config.image
    dogbreed = IMAGE_PATH.split('/')[-2]
    dogbreed_num = IMAGE_PATH.split('/')[-1].split('_')[-1].split('.')[0]

    # Define the model
    model = models.resnet152()
    model.fc = nn.Linear(2048, 120)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))['state_dict']
    new_dict = OrderedDict((key.replace('model.', ''), value) for key, value in state_dict.items())
    model.load_state_dict(new_dict)
    model.eval()

    # Define the preprocessing and postprocessing steps
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the image
    img_file = Image.open(IMAGE_PATH)

    # Preprocess the image
    img_tensor = transform(img_file)
    img_tensor = img_tensor.unsqueeze(0)

    # Get the predicted class from the model
    class_idx = sorted(os.listdir('./datasets_dogbreed/val/')).index(dogbreed)

    # Define the Grad-CAM object
    gradcam = GradCam_resnet(model, target_layer_names=["layer4"], target_sub_layer_names=["conv3"])

    # Generate the Grad-CAM for the predicted class
    gradcam_img = gradcam(img_tensor, index=class_idx)

    # Convert the heatmap to RGB and superimpose it on the original image
    img = cv2.imread(IMAGE_PATH)
    height, width, _ = img.shape

    heatmap = cv2.resize(gradcam_img, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    superimposed_img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)
    superimposed_img = superimposed_img / np.max(superimposed_img)
    superimposed_img = heatmap * 0.3 + superimposed_img * 0.5

    cv2.imwrite('Grad-CAM_images/{}_{}_Grad-CAM.jpg'.format(dogbreed, dogbreed_num), superimposed_img * 255)

if __name__ == '__main__':
    MODEL_PATH = './saved/models/DogBreed_ResNet152_Pretrained_Freeze_ColorJitter_lr0.005/0215_141830/model_best.pth'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/val/beagle/n02088364_13236.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/val/afghan_hound/n02088094_4219.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/train/eskimo_dog/n02109961_1276.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/train/eskimo_dog/n02109961_1076.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/train/siberian_husky/n02110185_8397.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/train/siberian_husky/n02110185_13127.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/test/eskimo_dog/n02109961_2492.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/test/eskimo_dog/n02109961_20013.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/test/eskimo_dog/n02109961_18009.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/test/eskimo_dog/n02109961_2317.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/test/siberian_husky/n02110185_5030.jpg'
    # IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/test/siberian_husky/n02110185_2614.jpg'
    IMAGE_PATH = '/home/yongchoooon/workspace/YCHPytorchTemplate/datasets_dogbreed/test/siberian_husky/n02110185_3651.jpg'
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-m', '--model', default=MODEL_PATH, type=str,
                      help='model file path')
    args.add_argument('-im', '--image', default=IMAGE_PATH, type=str,
                      help='image file path')

    config = args.parse_args()
    main(config)