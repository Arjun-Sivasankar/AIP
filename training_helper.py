#!/usr/bin/env python
# coding: utf-8


import numpy as np
import timeit
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from enum import Enum
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats


class Processor(str, Enum):
    riscv = 'riscv'
    # tensilica = 'tensilica'


class Program(str, Enum):
    bitcount = 'bitcount_MIBench'
    basicmath = 'basicmath_MIBench'
    fft = 'fft_MIBench'
    # gsm = 'gsm_MIBench'
    # jpeg = 'jpeg_MIBench'
    # stringSearch = 'stringSearch_MIBench'
    susan = 'susan_MIBench'
    qsort = 'qsort_MIBench'


class TestCase(str, Enum):
    training = 'training'
    test = 'test'


class run(str, Enum):
    run0 = 'run0'
    run1 = 'run1'
    run2 = 'run2'
    # run3 = 'run3'
    # run4 = 'run4'
    # run5 = 'run5'
    # run6 = 'run6'
    # run7 = 'run7'
    # run8 = 'run8'
    # run9 = 'run9'


class MemoryPortRiscv(str, Enum):
    # iram = 'iram_trace_xs.mem'
    dram = 'dram_trace_xs.mem'


class MemoryPortTensilica(str, Enum):
    iram = 'iram_trace_addr'
    dram = 'pif_trace_addr'


class MemoryPort(str, Enum):
    iram = "iram"
    dram = "dram"


class Model(str, Enum):
    cnn = "cnn"
    dense = "dense"
    LSTM = "LSTM"
    GRU = "GRU"
    attention_LSTM = "attention_LSTM_model"
    attention_dense = "attention_dense_model"
    transformer = "transformer_model"
    self_attention = "self_attention"
    cross_attention = "cross_attention"
    decoder = "decoder"
    attention = "attention"
    transformer_encoder = "transformer_encoder"


class chosen_models(str, Enum):  # other models should be added such as M3, M4...
    M1 = "M1"
    M2 = "M2"


def address_hash(x, shift, modulo):
    x = x >> shift
    x = x % (2 ** modulo)
    return x


def get_file_names(processor: Processor, program: Program, testcase: TestCase, run: run,
                   memory_port: MemoryPort, model_option: Model, feature_size, testing=False, models=None):
    if processor == Processor.riscv:
        riscv = True
        if memory_port == MemoryPort.dram:
            dataset = MemoryPortRiscv.dram
        else:
            dataset = MemoryPortRiscv.iram
    else:
        riscv = False
        if memory_port == MemoryPort.dram:
            dataset = MemoryPortTensilica.dram
        else:
            dataset = MemoryPortTensilica.iram
    if testcase == TestCase.test:
        file = "data_" + testcase + "/" + processor + "/" + program + "/" + run + "/" + dataset
    else:
        file = "data_" + testcase + "/" + processor + "/" + program + "/" + dataset
    file_preprocessed = ""
    model_name = ""
    if feature_size is not None:
        file_preprocessed = file + "_preprocessed" + "_" + str(feature_size[0]) + "_" + str(
            feature_size[1]) + "_" + str(feature_size[2])
        if testing:
            model_name = "result/model/chosen_models/" + str(processor) + "/" + str(program) + "/" + str(
                dataset) + "/" + str(feature_size[0]) + "_" + str(feature_size[1]) + "_" + str(
                feature_size[2]) + "_" + models
            if model_option == "cnn":
                model_name = model_name + "_cnn"
            elif model_option == "LSTM":
                model_name = model_name + "_lstm.h5"
            elif model_option == "transformer_model":
                model_name = model_name + "_trans"
        else:
            model_name = "result/model/" + str(processor) + "/" + str(program) + "/" + str(dataset) + "/" + str(
                model_option) + str(feature_size[0]) + "_" + str(feature_size[1]) + "_" + str(feature_size[2])
    return riscv, file, file_preprocessed, model_name


def data_import(input_file, max_rows=None, riscv=False):
    if riscv:
        pc, address, timestamp = np.loadtxt(input_file, delimiter=" ", skiprows=0, usecols=(2, 3, 0),
                                            max_rows=max_rows, dtype=np.uint32,
                                            converters={_: lambda s: int(s, 16) for _ in range(4)}, unpack=True)
        interval = np.zeros(len(timestamp))
        for i in range(0, len(timestamp) - 1):
            interval[i] = timestamp[i + 1] - timestamp[i]
    else:
        address, interval = np.loadtxt(input_file, delimiter=";", skiprows=2, usecols=(0, 1),
                                       max_rows=max_rows, dtype=np.uint32, unpack=True)
    print("data entries: ", "{:e}".format(len(address)))
    print("### data loaded ###")
    return address, interval, pc


def one_hot_encoder(interval):
    one_hot_code = np.zeros((32,), dtype=int)
    one_hot_code[interval] = 1
    return one_hot_code


def data_kernel(i, address, interval, pc, address_num, interval_num, pc_num):
    if address_num == 0 and pc_num == 0:
        feature = interval[i:i + interval_num]
    elif address_num == 1 and pc_num == 0:
        feature = np.concatenate((interval[i:i + interval_num], [address[i + interval_num]]), axis=None)
    elif address_num == 0 and pc_num == 1:
        feature = np.concatenate((interval[i:i + interval_num], [pc[i + interval_num]]), axis=None)
    elif address_num == 1 and pc_num == 1:
        feature = np.concatenate((interval[i:i + interval_num], [pc[i + interval_num]], [address[i + interval_num]]),
                                 axis=None)
    elif address_num == interval_num:
        feature = np.dstack((address[i: i + interval_num], interval[i:i + interval_num]))
    elif pc_num == interval_num:
        feature = np.dstack((pc[i: i + interval_num], interval[i:i + interval_num]))
    label = interval[i + interval_num]
    return feature, label


def data_preprocessing(address, interval, pc, hash_shift, hash_modulo, interval_max, data_size, dummy=False,
                       Trans=False):
    address = np.vectorize(address_hash)(address, hash_shift, hash_modulo) if not dummy else np.arange(2000)
    pc = np.vectorize(address_hash)(pc, hash_shift, hash_modulo) if not dummy else np.arange(2000)
    interval = np.vectorize(min)(interval, interval_max - 1)
    print("### addresses and pc hashed ###")

    address_num = data_size[0]
    interval_num = data_size[1]
    pc_num = data_size[2]
    total_data_set_size = len(address) - interval_num
    if address_num != interval_num and pc_num != interval_num:
        features = np.zeros((total_data_set_size, address_num + interval_num + pc_num), dtype=np.float32)

    else:
        features = np.zeros((total_data_set_size, interval_num, 2), dtype=np.float32)

    if Trans:
        label = np.zeros((total_data_set_size, interval_num), dtype=np.float32)
    else:
        label = np.zeros((total_data_set_size, 1), dtype=np.float32)

    start_time = timeit.default_timer()
    for i in range(0, len(address) - interval_num):
        features[i], label[i] = data_kernel(i, address, interval, pc, address_num, interval_num, pc_num)

    print(f'time for preprocessing: {timeit.default_timer() - start_time:.3f}', flush=True)
    print("### data preprocessed ###", flush=True)
    print(features.shape)
    return features, label


def data_preprocessed_store(file_name, features, label):
    np.savez(file_name, features=features, label=label)


def data_split(input_data, label, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(input_data, label, test_size=test_size,
                                                        random_state=random_state)
    print("### data split ###")
    # print(X_train.shape)
    # print(y_train.shape)
    return X_train, X_test, y_train, y_test


# probability function and average access value
def interval_distribution(interval):
    # variables
    interval_length = interval.size
    max_value = np.max(interval)
    print(max_value)
    distribution = np.zeros(max_value + 1)

    for i in range(interval_length):
        distribution[interval[i]] += 1
    probability = distribution / interval_length
    mu = np.sum(np.arange(max_value + 1) * probability)

    return probability, mu


def histogram(processor: Processor, program: Program, testcase: TestCase, memory_port: MemoryPort, model, feature_size,
              Trans, max_rows=None,
              plot=False):
    # file names
    runterm = None
    riscv, file, file_preprocessed, model_name = get_file_names(processor, program, testcase, runterm, memory_port,
                                                                model, feature_size)
    address, interval, pc = data_import(file, max_rows=max_rows, riscv=riscv)

    # calculate distribution
    density, average = interval_distribution(interval)
    print("average interval size:", round(average, 2))

    # plot
    if plot:
        plt.plot(density, marker='o')
        plt.grid()
        plt.title(processor + "/" + program + "/" + testcase + "/" + memory_port)
        plt.show()

    return density


def correlation(riscv, file, hash_shift, hash_modulo, interval_max):
    address, interval = data_import(file, max_rows=None, riscv=riscv)
    address = np.vectorize(address_hash)(address, hash_shift, hash_modulo)
    interval = np.vectorize(min)(interval, interval_max - 1)
    print("### addresses hashed ###")

    print(interval.shape)

    autocorrelation = np.array([])

    delay = np.array([0])
    coeff = np.equal(interval, interval)
    coeff = np.sum(coeff)
    autocorrelation = np.append(autocorrelation, coeff)

    for shift in range(1, 21):
        coeff = np.equal(interval[:-shift], interval[shift:])
        coeff = np.sum(coeff)
        # print(coeff)
        autocorrelation = np.append(autocorrelation, coeff)
        delay = np.append(delay, shift)
    plt.plot(delay, autocorrelation, marker='o')
    plt.xticks(range(1, 21))
    plt.title(file + " autocorrelation")
    plot_acf(x=interval, lags=20, fft=True, title="interval autocorrelation(original)")
    plt.show()
