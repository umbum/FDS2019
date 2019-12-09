import os
import pathlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
POINT_COUNT_BOUND = 6000    # 너무 많으면 다 시각화 하는 것도 불가능하고 t-SNE 학습이 너무 오래 걸린다.

datasets = ["bs140513"]
iterations_ls = [250, 500, 750, 1000]
perplexity_ls = [3, 10, 30, 50, 100]
pca_dim_ls = [25]
learning_rate_ls = [10, 50, 100, 200]

def generate_embedding(
    dataset, iterations, perplexity, pca_dim, learning_rate, verbose=1, mode="two_files"
):
    path = f"embeddings/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}"

    def display(string):
        if verbose:
            print(string)

    if os.path.exists(path):
        if os.path.exists(path + f"/data.csv"):
            display(path + " already exists.")
            return
    else:
        os.makedirs(path)

    if mode == "two_files":
        data = pd.read_csv(DATA_PATH.joinpath(f"{dataset}_input.csv"))
        labels = pd.read_csv(DATA_PATH.joinpath(f"{dataset}_labels.csv"))
        clustering_result = pd.read_csv(DATA_PATH.joinpath(f"{dataset}_clustering_result.csv"))
    elif mode == "one_file":
        data = pd.read_csv(
            DATA_PATH.joinpath(f"{dataset}.csv"), index_col=0, encoding="ISO-8859-1"
        )
        labels = data.index
    else:
        assert "two_files || one_file"
    
    data = data[:POINT_COUNT_BOUND]
    labels = labels[:POINT_COUNT_BOUND]
    clustering_result = clustering_result[:POINT_COUNT_BOUND]
    nb_col = data.shape[1]

    pca = PCA(n_components=min(nb_col, pca_dim))
    data_pca = pca.fit_transform(data.values)

    tsne = TSNE(
        n_components=3,
        n_iter=iterations,
        learning_rate=learning_rate,
        perplexity=perplexity,
        random_state=1131,
    )

    embedding = tsne.fit_transform(data_pca)

    embedding_df = pd.DataFrame(embedding, columns=["x", "y", "z"])

    embedding_df.index = np.squeeze(labels.values)

    embedding_df["cluster"] = np.squeeze(clustering_result.values)

    embedding_df.to_csv(path + f"/data.csv")

    display(f"{path} has been generated.")


from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=8) as executor:
    q = []
    for dataset in datasets:
        for iterations in iterations_ls:
            for perplexity in perplexity_ls:
                for pca_dim in pca_dim_ls:
                    for learning_rate in learning_rate_ls:
                        args = [dataset, iterations, perplexity, pca_dim, learning_rate]
                        kwargs = dict(mode='two_files',)
                        q.append( executor.submit(generate_embedding, *args, **kwargs) )
                        # generate_embedding(
                        #     dataset,
                        #     iterations,
                        #     perplexity,
                        #     pca_dim,
                        #     learning_rate,
                        #     mode="two_files",
                        # )

