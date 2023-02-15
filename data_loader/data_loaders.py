from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from base import BaseDataLoader
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import torch


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = "data/"
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class NotMnistDataLoader(BaseDataLoader):
    """
    NotMNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = NotMnistDataset(training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class NotMnistDataset(TensorDataset):
    """
    NotMNIST dataset using TensorDataset
    """
    def __init__(self, training = True):
        dataset_x_list = []
        dataset_y_list = []
        if training:
            DATASET_PATH = "datasets_notmnist/train"
        else:
            DATASET_PATH = "datasets_notmnist/test"

        for folder in tqdm(Path(DATASET_PATH).iterdir()):
            alphabet = str(folder)[-1]

            for alpha in tqdm(folder.iterdir()):
                try:
                    im = Image.open(str(alpha))
                except:
                    continue
                
                im_tensor = transforms.ToTensor()(im)
                dataset_x_list.append(im_tensor)
                dataset_y_list.append(ord(alphabet) - 65) # A => 0, B => 1, ... , J => 9

        self.tensor_x = torch.stack(dataset_x_list)
        self.tensor_y = torch.tensor(dataset_y_list)

        self.tensors = (self.tensor_x, self.tensor_y)

class DogBreedDataLoader(BaseDataLoader):
    """
    DogBreed data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, validation=False):
        
        if training:
            DATASET_PATH = "datasets_dogbreed/train"
            trsfm = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
            if validation:
                DATASET_PATH = "datasets_dogbreed/val"
                trsfm = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])])
        else:
            DATASET_PATH = "datasets_dogbreed/test"
            trsfm = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

        self.dataset = datasets.ImageFolder(DATASET_PATH, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
