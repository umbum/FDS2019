# sckit-learn

from sklearn.cluster import KMeans, DBSCAN
from autoencoder import data_load
import pandas as pd
import os
import pprint
import collections

################
#  data params #
################
base_path = "../data"
base_csv_file = "bs140513"
answer_file_path =  os.path.join(base_path, f"{base_csv_file}_origin.csv")
data_file_path = os.path.join(base_path, f"{base_csv_file}_input.csv")
result_file_path = os.path.join(base_path, f"{base_csv_file}_{{}}_clustering_result.csv")

data = pd.read_csv(data_file_path).to_numpy()
answer_data = data_load.source_csv_load(answer_file_path, "fraud")[1]

for n_clusters in range(2,10 +1):
    cluster = KMeans(n_clusters=n_clusters)
    cluster.fit(data)

    indexer = lambda lis, val : set([i for i, v in enumerate(lis) if v == val])
    label_ind = indexer(answer_data, 1)

    # 클러스터링 결과 파일에 출력
    with open(result_file_path.format(n_clusters), 'wt') as wf:
        print('result', file=wf)
        print(*cluster.labels_, sep='\n', file=wf)

    # 클러스터의 포인트 갯수 10000개를 기준으로 fraud여부를 나눔.
    divide_level = 10000
    resulting_fraud = set()

    print(f'{n_clusters} 클러스터 정보 출력  {len(label_ind)}')
    # 클러스터링 별 각각 정보출력
    for i in range(n_clusters):
        points = indexer(cluster.labels_, i)
        corr_size = len(points & label_ind)
        acc = corr_size / len( points )
        size = len(points)
        # print(f"{i} 정확도 {acc:0.4f} fraud갯수/클러스터크기 {corr_size}/{size}")

        if size < divide_level:
            resulting_fraud |= points
    fraud_size, total_size = len(resulting_fraud), len(label_ind)
    corr_size = len(resulting_fraud & label_ind)
    wrong_size = len(resulting_fraud - label_ind)

    print(f"{divide_level}개 포인트 기준 fraud 정확도 : \t{corr_size:4d}/{total_size:4d} = \t{corr_size/total_size:0.4f}")
    print(f"{divide_level}개 포인트 기준 fraud FP 비율 : \t{wrong_size:4d}/{fraud_size:4d} = \t{wrong_size/fraud_size:0.4f}\n")