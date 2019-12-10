import pandas as pd
import hashlib
import functools

################
# load file    #
################


# 같은 설정에 대해서는 데이터를 유지합니다.
@functools.lru_cache()
def source_csv_load(data_csv_file, answer_label, as_numpy=True, needs_convert=True):
    '''
    내부적으로 해싱을 진행합니다.
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

    converted = source_data.applymap(str_convert) if needs_convert else source_data
    if as_numpy:
        return converted.to_numpy(), answer_data.to_numpy(), source_data.columns
    else:
        return converted, answer_data, source_data.columns
