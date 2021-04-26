# LandmarkDetector
창의적공학설계 2021 봄학기 프로젝트 리포지토리


## environment requirement

anaconda 사용
https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html
requirement.txt를 이용해서 새 environment 구성하는 것을 추천합니다.

### VS code가 제대로 된 python kernel을 선택하지 못하는 문제
Anaconda navigator에서 제대로 된 환경을 고른 뒤 vs code를 프로그램 내에서 실행하면 됨.

### GIT-LFS
git lfs를 사용해서 모델을 올렸습니다. git lfs를 설치해서 써야 할 수도 있음.

### cv2.imread()가 사진을 못 불러오는 경우
경로와 파일명 중에 한글이 포함되어 있으면 제대로 읽어오지 못하므로, 파일명을 잘 바꿔줘야 한다.
```bash
rename -- 's/[^0-9a-zA-Z.-]/_/g' *
```
이걸로 non-ascii 이름을 다 날려버리는 방법이 있다. linux / WSL에서 할 것

### 중요! "AutoAlign" 폴더에 필요한 사진들과 데이터를 넣으십시오!
말 그대로 혹시 몰라서 로컬에만 저장하고 git에는 안 올렸음. 나중에 다른 파일들도 bfg-cleaner로 날려버려야 할 수도 있습니다. 따로 넣어주면 잘 작동합니다.

## HOWTO
### Training
test.ipynb의 마지막 직전 block까지 모두 실행시키면 됨.

### Testing
test.ipynb의 마지막 블록을 작동하면 됨.

### Using pretrained model
face_landmarks.pth를 test.ipynb의 마지막 블록을 참고해서 잘 이용하면 됩니다.

### 결과
뭐 그럭저럭 나옴. loss ~= 20 정도로 오차가 꽤 존재하는 편이다. Ceph landmark 22개 좌표를 모두 return하며 format은 tsv에서 위에서 아래로 읽은 것 중에 특정 부분에 해당함.

## TODO
지금은 정확도가 떨어진다.

### 데이터 튜닝
- 지금은 grayscale에서 train중인데 RGB를 모두 이용하게 해서 정확도를 높이기.
- hyperparameter 조정
- pretrained model으로 얼굴만 미리 크롭해서 트레이닝하기. 실제 활용시에도 얼굴을 크롭해서 넣어주는 방식으로 활용 가능하다.
- 랜덤 회전, 밝기 조절등 transform을 추가해서 training sample 뻥튀기하기 (500여개 --> 1000+, 가능하면 10000+까지)

### 모델
- Resnet-16으로 충분한지 체크하고, 그렇지 않으면 더 깊은 model로 바꾸기
- 논문 참고해서 가장 state-of-the-art model로 갈아탈 수도 있음

### Training
- epoch를 더 늘려보기(사실 근본적인 문제 해결 없이는 어느 정도 선에서 안 줄어들 것 같음)
- lr을 0.0001로 줄여보기(시간이 10배 더 걸리겠지만 후반부 막장상황 해결에 도움될듯)
- 왜 초반 loss가 엄청 크게 나오는지 생각해보기
- transfer가 가능한가? (이건 불가능할듯)
