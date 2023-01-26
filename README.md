# NotMnist
```Plain text
딥러닝 모델에 NotMnist 데이터셋을 학습시키는 Customized Pytorch template
```
- [Pytorch Template Github](https://github.com/victoresque/pytorch-template)

## Contents
- `dataset handling`, `data loader`, `model`, `trainer`, `logger` 등 DL 학습을 위한 코드 구조 이해
- 원하는 `dataset`을 불러와서 학습에 활용
- 나만의 `data loader`, `model` 코드를 작성하기
- 학습을 진행하고 `logging`, `tensorboard writing` 등 기록 및 저장 과정 익히기
- 학습 과정 중 `slack`으로 알림 보내는 기능 입히기
  - [slack_alarm.py](./slack_alarm.py)