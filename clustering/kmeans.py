# sckit-learn

from sklearn.cluster import KMeans
from autoencoder import data_load
import pandas as pd
import os
import pprint
import collections

################
#  data params #
################
base_path = "../data"
base_csv_file = "bs140513_032310_striped"
answer_file_path =  os.path.join(base_path, f"{base_csv_file}.csv")
data_file_path = os.path.join(base_path, f"{base_csv_file}_autoencoded.csv")
result_file_path = os.path.join(base_path, f"{base_csv_file}_clustering_result.csv")

data = pd.read_csv(data_file_path).to_numpy()
answer_data = data_load.source_csv_load(answer_file_path, "fraud")[1]

n_clusters=6
cluster = KMeans(n_clusters=n_clusters)
cluster.fit(data)

indexer = lambda lis, val : set([i for i, v in enumerate(lis) if v == val])
label_ind = indexer(answer_data, 1)

# 클러스터링 결과 파일에 출력
with open(result_file_path, 'wt') as wf:
    print('result', file=wf)
    print(*answer_data, file=wf)

print('클럿트 정')
# 클러스터링 별 각각 정보출
for i in range(n_clusters):
    points = indexer(cluster.labels_, i)
    acc = len(points & label_ind) / len( points )
    size = len(points)
    print(f"{i} 정확도 {acc:0.4f} 크기 {size}")
