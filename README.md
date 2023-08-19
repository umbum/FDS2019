# Fraud Detection & Visualization
<img width="219" alt="image" src="https://github.com/umbum/FDS2019/assets/25507015/9723a300-bc73-4df5-9990-c529f298fad8">
<img width="452" alt="image" src="https://github.com/umbum/FDS2019/assets/25507015/04a6cb5a-fdeb-4e4f-89ce-637da1109e49">


# Installation and Getting Started
#### Install
- python3
```bash
$ git clone https://github.com/umbum/FDS2019
$ cd FDS2019
$ pip3 install -r requirements.txt
```

#### Clustering

##### 데이터 전처리
* autoencoder 폴더에서 작업
    1. params.py의 n_inputs를 입력 데이터 차원에 맞추고, 아래 두 실행파일에서 데이터 파일이름 입력.
    2. `python3 train.py`를 통해 오토인코더 학습 모델 생성.
    3. `python3 data_encodde.py`를 통해서 오토인코딩한 데이터 생성.

##### 클러스터링 진행
* clustering 폴더에서 작업
    1. `python3 kmeans.py`를 실행시켜 클러스터링 레이블 생성

#### Run
```bash
// 시각화 데이터 생성. POINT_COUNT_BOUND=5000이면 약 10분 소요.
$ python3 generate_embeddings.py
// 실행
$ python3 app.py
```

#### Dataset Naming convention
```
/data/{dataset_name}_origin.csv               # original  dataset (시각화 과정에 사용됨)
/data/{dataset_name}_numerical.csv            # 문자열 필드를 hash하여 실수로 변경한 dataset (없어도 상관 없음)
/data/{dataset_name}_input.csv                # autoencoded dataset (시각화 과정 input으로 들어감)
/data/{dataset_name}_labels.csv               # label data
/data/{dataset_name}_clustering_result.csv    # clustering result label data
```

#### N-d dataset -> 3-d dataset으로 변환
- 먼저 `generate_embeddings.py` 파일을 열어서 datasets 항목에 데이터셋 추가해야 함
- 데이터셋은 `/data` 폴더에 위치해야 함
- PCA, t-SNE를 거쳐 column이 `(answer, x, y, z, cluster)`인 데이터 생성
- 3-d dataset은 `/embeddings` 폴더에 생성됨
``` 
$ python3 generate_embeddings.py
```
