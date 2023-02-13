# YCHPytorchTemplate
```Plain text
딥러닝 모델에 데이터셋을 학습시키는 Customized Pytorch template
```
- [Pytorch Template Github](https://github.com/victoresque/pytorch-template)

## Contents
- `dataset handling`, `data loader`, `model`, `trainer`, `logger` 등 DL 학습을 위한 코드 구조 이해
- 원하는 `dataset`을 불러와서 학습에 활용
- 나만의 `data loader`, `model` 코드를 작성하기
- 학습을 진행하고 `logging`, `tensorboard writing` 등 기록 및 저장 과정 익히기
- 학습 과정 중 `slack`으로 log 알림 보내는 기능 추가
  - [slack_alarm.py](./slack_alarm.py)

## Datasets
- [NotMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)
- [Stanford Dog Breed](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

## Assignment
### Dog Breed Classification
- Notion: [Dog Breed Classification](https://www.notion.so/Dog-Breed-adff4af9047f46c481a80b22c14ca397)
  - DensetNet 코드 구현 및 학습
  - VGG 모델 학습 이후 Generation 기법 적용해 성능 올리기 연습
  - Transfer Learning으로 성능 올리기