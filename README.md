# LandmarkDetector
창의적공학설계 2021 봄학기 프로젝트 리포지토리

## HOWTO

### Setting environment
`conda env create -f requirement.txt` 이후 `pip install -r requirement_pip.txt`

### Training
train.ipynb의 처음 block부터 "From Face detection to landmark detection, IRL" 부분 이전까지 실행해주세요. 이 때 데이터셋 경로를 파일 내에서 변경해주어야 합니다.

### Testing
inference.ipynb를 실행해주세요. 이 때 파일 내에서 모델 경로(`model_path`)와 이미지 경로(`photo_image_path` 등)를 잘 설정하여야 합니다. 혹은 parse.py를 실행해주세요.

### Using pretrained model
모델 자체는 공개하지 않습니다.

### Running program 
parse.py를 CUI에서 실행하면 됩니다. `python parse.py -h` 를 참고하여 실행해주세요.

### python parse.py -h

```bash
  Usage: python parse.py [options]

  아무 옵션도 설정하지 않을 경우 도움말 메시지가 출력됩니다.
  -h, --help            도움말 메시지를 출력하고 종료합니다.
  -v, --verbose         랜드마크 정합 결과를 화면에 출력하고, 정합 이미지를 저장합니다. 단계별 실행 시간도 출력합니다.
  -t, --test            측면 이미지에서 랜드마크만 인식하고 출력한 뒤 종료합니다.
  -fi FILM_IMAGE, --film_image FILM_IMAGE
                        필요한 경우, 필름 이미지의 경로를 지정합니다. (기본: film.jpg)
  -ft FILM_TSV, --film_tsv FILM_TSV
                        필름 랜드마크 파일의 경로를 지정합니다. (기본: film.txt)
  -pi PHOTO_IMAGE, --photo_image PHOTO_IMAGE
                        측면 사진 이미지의 경로를 지정합니다. (기본: photo.jpg)
  -pt PHOTO_TSV, --photo_tsv PHOTO_TSV
                        필요한 경우, 측면 사진 랜드마크의 경로를 지정합니다. 테스트용으로, 실 사용시에는 지정할 필요 없습니다.
                        (기본: photo.txt)
  -m MODEL, --model MODEL
                        tar 형태로 된 모델의 경로를 지정합니다. (기본: model.tar)
  -o OUTPUT, --output OUTPUT
                        출력할 정합 텍스트 파일의 경로를 지정합니다. (기본: result.txt)
  --output_image OUTPUT_IMAGE
                        verbose 옵션이 지정되었을 경우 저장할 정합 이미지의 경로를 지정합니다.
                        경로가 지정되지 않으면 이미지를 저장하지 않습니다.

  도움말 메시지를 출력하려면
    > python parse.py -h
  이미지 정합을 하려면
    > python parse.py -m PATH_TO_MODEL -ft PATH_TO_FILM_TSV -pi PATH_TO_PHOTO_IMAGE
  정답 랜드마크와 비교한 정합 이미지를 출력하려면
    > python parse.py -v -m PATH_TO_MODEL -fi PATH_TO_FILM_IMAGE -ft PATH_TO_FILM_TSV -pi PATH_TO_PHOTO_IMAGE -pt PATH_TO_PHOTO_TSV


```

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
Anaconda navigator을 사용해 개발하였으며 requirement.txt를 이용해서 새 environment를 구성하는 것을 추천합니다. `conda env create -f requirement.txt` 이후 `pip install -r requirement_pip.txt`

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

## Copyrights

모델의 training에 오스템 사의 측면 사진 데이터가 이용되었습니다.

- Used facenet-pytorch: it follows MIT license. https://github.com/timesler/facenet-pytorch 
- Used part of nuitka: it follows Apache 2.0 license. https://github.com/Nuitka/NUITKA-Utilities/tree/master/hinted-compilation
- For full list of programs used, please refer to requirement.txt and requirement_pip.txt
