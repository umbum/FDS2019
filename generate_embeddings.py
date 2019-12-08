import os
import pathlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

# datasets = ["wikipedia_3000", "twitter_3000", "crawler_3000"]
# iterations_ls = [250, 500, 750, 1000]
# perplexity_ls = [3, 10, 30, 50, 100]
# pca_dim_ls = [25, 50, 100]
# learning_rate_ls = [10, 50, 100, 200]
datasets = ["bs140513_032310_striped_hashed"]
iterations_ls = [250, 500, 750]
perplexity_ls = [3, 10, 30]
pca_dim_ls = [25]
learning_rate_ls = [10, 50, 100]


def generate_embedding(
    dataset, iterations, perplexity, pca_dim, learning_rate, verbose=1, mode="two_files"
):
    path = f"demo_embeddings/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}"

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
    elif mode == "one_file":
        # data = pd.read_csv("/Users/umbum/source/dash-sample-apps/apps/dash-tsne/data/bs140513_032310_striped_hashed.csv", index_col=0, encoding="ISO-8859-1")
        data = pd.read_csv(
            DATA_PATH.joinpath(f"{dataset}.csv"), index_col=0, encoding="ISO-8859-1"
        )
        labels = data.index
    else:
        assert "two_files || one_file"

    data, labels = data[:300], labels[:300]  # 너무 많으면 다 시각화 하는 것도 불가능하고 t-SNE 학습이 너무 오래 걸린다.
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

    embedding_df.to_csv(path + f"/data.csv")

    display(f"{path} has been generated.")


for dataset in datasets:
    for iterations in iterations_ls:
        for perplexity in perplexity_ls:
            for pca_dim in pca_dim_ls:
                for learning_rate in learning_rate_ls:
                    generate_embedding(
                        dataset,
                        iterations,
                        perplexity,
                        pca_dim,
                        learning_rate,
                        mode="two_files",
                    )
