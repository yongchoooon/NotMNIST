import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from collections import OrderedDict
import torch
import torchvision.models as models

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MyDenseNetModel(BaseModel):
    def __init__(self, num_classes = 120, growth_rate = 32, num_network_layers = 121):
        super().__init__()
        self.num_first_input_features = 2 * growth_rate
        
        if num_network_layers == 121:
            self.nums_blocks = (6, 12, 24, 16)
        elif num_network_layers == 169:
            self.nums_blocks = (6, 12, 32, 32)
        elif num_network_layers == 201:
            self.nums_blocks = (6, 12, 48, 32)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(2 * growth_rate)),
            ('relu0', (nn.ReLU(inplace=True))),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_input_features = self.num_first_input_features
        for b in range(len(self.nums_blocks)):
            block = DenseBlock(
                num_layers = self.nums_blocks[b],
                num_input_features = num_input_features,
                growth_rate = growth_rate
            )
            self.features.add_module(f"DenseBlock{b}", block)
            num_input_features = num_input_features + self.nums_blocks[b] * growth_rate
            
            if b != len(self.nums_blocks) - 1:
                num_output_features = num_input_features // 2
                trans = TransitionLayer(num_input_features, num_output_features)
                self.features.add_module(f"TransmisionLayer{b}", trans)
                num_input_features = num_output_features

        # BN, ReLU
        self.features.add_module('last_norm', nn.BatchNorm2d(num_input_features))
        self.features.add_module('last_ReLU', nn.ReLU(inplace = True))

        # 120D fc
        self.classifier = nn.Linear(num_input_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    def forward(self, input):
        output = self.features(input)
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = torch.flatten(output, 1)
        output = self.classifier(output)
        return F.log_softmax(output, dim=0)


class DenseBlock(nn.ModuleList):
    def __init__(self, num_layers, num_input_features, growth_rate):
        super().__init__()
        for l in range(num_layers):
            layer = DenseLayer(num_input_features = num_input_features + l * growth_rate,
                               growth_rate = growth_rate)
            self.append(layer)

    def forward(self, input):
        features = []
        features.append(input)
        for layer in self:
            output = layer(torch.cat(features, 1))
            features.append(output)
        return torch.cat(features, 1)



class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate):
        super().__init__()
        self.norm0 = nn.BatchNorm2d(num_input_features)
        self.relu0 = nn.ReLU(inplace = True)
        self.conv0 = nn.Conv2d(in_channels = num_input_features, out_channels = 4 * growth_rate, kernel_size = 1, stride = 1, padding = 0)
        self.norm1 = nn.BatchNorm2d(4 * growth_rate)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(in_channels = 4 * growth_rate, out_channels = growth_rate, kernel_size = 3, stride = 1, padding = 1)


class TransitionLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels = num_input_features, out_channels = num_output_features, kernel_size = 1, stride = 1, padding = 0)
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

class DenseNetModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Linear(1024, 120)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=0)

class ResNet152Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = models.resnet152(pretrained=False)
        self.model.fc = nn.Linear(2048, 120)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=0)

class VGG19Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = models.vgg19(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 120)
        )

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=0)

class VGG19BNModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = models.vgg19_bn(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 120)
        )

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=0)