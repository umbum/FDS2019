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
result_file_path = os.path.join(base_path, f"{base_csv_file}_label.csv")

data = pd.read_csv(data_file_path).to_numpy()
answer_data = data_load.source_csv_load(answer_file_path, "fraud")[1]

cluster = KMeans(n_clusters=2)
cluster.fit(data)
fraud_ind = lambda lis : [ i for i,v in enumerate(lis) if v == 1  ]
a,p = [ set(fraud_ind(elem)) for elem in [cluster.labels_, answer_data] ]

pr = collections.OrderedDict(
    answer_size = len(a),
    predict_size = len(p),
    correct_size = len(a & p),
    total_size = len(a | p),
    wrong_size = len(a | p) - len(a & p),
)

with open(result_file_path, 'wt') as wf:
    print('label', file=wf)
    print(*cluster.labels_, sep='\n', file=wf)

pprint.pprint(pr)