from training_helper import *

dummy_data = False


def training_preprocessing (file,file_preprocessed,  riscv, feature_size,a ,b, interval_max ):
    
    # load data

    data = data_import(file, max_rows=None, riscv=riscv)
    
    # preprocess data
    input_data, label = data_preprocessing(data[0], data[1], data[2], a, b, interval_max, feature_size, dummy=dummy_data)

    # store preprocessed data
    data_preprocessed_store(file_preprocessed, input_data, label)

