import pandas as pd
import hashlib

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

    hash_func = hashlib.sha1
    hash_max = int("f"*40, 16)

    source_data = data[source_columns]
    answer_data = data[answer_columns]

    def str_convert(elem):
        if isinstance(elem, str):
            hashed = hash_func(elem.encode()).hexdigest()
            return int(hashed, 16) / hash_max
        else:
            return elem

    converted = source_data.applymap(str_convert)
    return converted.to_numpy(), answer_data.to_numpy()
