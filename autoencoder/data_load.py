import pandas as pd

################
# load file    #
################


def csv_load(data_csv_file, answer_label):
    '''
    return train_x, train_y
    as numpy_ndarray
    '''

    data = pd.read_csv(data_csv_file)
    answer_columns = answer_label
    source_columns = list( data.columns )
    source_columns.remove(answer_label)

    source_data = data[source_columns].to_numpy()
    answer_data = data[answer_columns].to_numpy()
    return source_data, answer_data
