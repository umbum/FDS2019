

#### Install
- python3
```bash
$ git clone https://github.com/umbum/FDS2019
$ cd FDS2019
$ pip3 install -r requirements.txt
```

#### Run
```bash
$ python3 app.py
```

#### Dataset Naming convention
```
/data/{dataset_name}_origin.csv               # original  dataset (문자열 필드 등.)
/data/{dataset_name}_input.csv                # numerical dataset
/data/{dataset_name}_labels.csv               # label data
/data/{dataset_name}_clustering_result.csv    # clustering result label data
```

#### N-d dataset -> 3-d dataset으로 변환
- 먼저 `generate_embeddings.py` 파일을 열어서 datasets 항목에 데이터셋 추가해야 함
- 데이터셋은 `/data` 폴더에 위치해야 함
- PCA, t-SNE를 거쳐 column이 `(label, x, y, z)`인 데이터 생성
- 3-d dataset은 `/embeddings` 폴더에 생성됨
``` 
$ python3 generate_embeddings.py
```