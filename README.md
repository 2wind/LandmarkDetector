# LandmarkDetector
창의적공학설계 2021 봄학기 프로젝트 리포지토리

## HOWTO

### Training
train.ipynb의 처음 block부터 "From Face detection to landmark detection, IRL" 부분 이전까지 실행해주세요.

### Testing
train.ipynb의 "From Face detection to landmark detection, IRL"을 실행해주세요. 이 때 파일 내에서 모델 경로와 이미지 경로를 잘 설정하여야 합니다.

### Using pretrained model
모델 자체는 공개하지 않습니다.

### 결과
MSE loss = 0.0001 정도까지 Train이 가능합니다. Test set(224x224) 에 대해, 평균 13px 정도의 오차가 나게 됩니다.

## More

### 데이터 튜닝
- Grayscale data, 224x224
- hyperparameter lr = 0.001, 
- pretrained model으로 얼굴만 미리 크롭해서 트레이닝하기. 실제 활용시에도 얼굴을 크롭해서 넣어주는 방식으로 활용 가능하다.
- 랜덤 회전, 밝기 조절등 transform을 추가해서 training sample 뻥튀기하기 (500여개 --> 1000+, 가능하면 10000+까지)

### 모델
- Resnext-50을 사용하다가, VRAM 문제로 Resnet-50 모델을 이용
- conv1, fc 레이어를 교체해서 사용
- 모델 전체에 대해 튜닝
- 가능하면 전결합층만 학습 후 모델 전체에 대해 학습시키는 것을 권장합니다

### Training
- 100 epoch 만으로도 충분히 좋은 결과가 나오며, epoch를 더 늘려서 학습도 가능
- overfitting 없음
- transfer learning 사용

## Troubleshooting
### environment requirement
Anaconda navigator 사용, requirement.txt를 이용해서 새 environment 구성하는 것을 추천합니다. `onda env create -f requirement.txt` 이후 `pip install -r requirement_pip.txt`

### VS code가 제대로 된 python kernel을 선택하지 못하는 문제
Anaconda navigator에서 제대로 된 환경을 고른 뒤 vs code를 프로그램 내에서 실행하면 됩니다.

### cv2.imread()가 사진을 못 불러오는 경우
경로와 파일명 중에 한글이 포함되어 있으면 제대로 읽어오지 못하므로, 파일명을 잘 바꿔줘야 합니다. 아래 명령은 bash에서 non-ascii 이름을 모두 underscore로 대체합니다.
```bash
rename -- 's/[^0-9a-zA-Z.-]/_/g' *
```

### 중요! "AutoAlign" 폴더에 필요한 사진들과 데이터를 넣으십시오!
gitignore 파일에 추가했으며, 실제 개발에는 해당 폴더에 NDA에 포함되는 이미지 파일들을 포함시켰습니다.

### 빈 커밋들이 매우 많습니다
리포지토리를 공개하기 위해 NDA에 위반되는 파일들을 BFG-repo cleaner로 제거했습니다. jupyter notebook에 들어간 이미지 파일들이 많았기 때문에, 어쩔 수 없이 파일들도 제거해야 했습니다.

