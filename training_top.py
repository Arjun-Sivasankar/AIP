#!/usr/bin/env python
import model
from training_helper import *
from training_preprocessing import *
from training_test_offline import *
from training_quasi_online import *
import tensorflow as tf
import os
import sys
import logging

logging.getLogger("tensorflow").setLevel(logging.DEBUG)
# import tensorflow_model_optimization as tfmot

gpus = tf.config.list_physical_devices('GPU')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print("Num GPUs Available: ", len(gpus))
print(tf.version.VERSION)
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# parameter

training = True  # training model and evaluation
testing = False  # only evaluation on several datasets
flt16 = False  # float 16-bit quantization
int8 = False  # integer 8 bit quantization
prune = False  # pruning model (might unfinished)

sys_num_addr = sys.argv[1]
sys_num_inter = sys.argv[2]
sys_num_pc = sys.argv[3]
sys_model = sys.argv[4]
sys_program = sys.argv[5]

feature_size = [int(sys_num_addr), int(sys_num_inter), int(sys_num_pc)]  # [addr_num, interval_num, pc_num]
print(int(sys_num_addr), int(sys_num_inter), int(sys_num_pc))
test_percentage = 0.33
hash_shift = 2
hash_modulo = 5
dummy_data = False
interval_max = 32
learning_rate_offline = 1e-3
# file name

processor = Processor.riscv
program = str(sys_program) + '_MIBench'  # Program.bitcount
memory_port = MemoryPort.dram
model_option = str(sys_model)  # "transformer_encoder"

if training:
    average_acc = 0
    with open("accuracy_evaluation.txt", "a") as hs:
        hs.write(
            str(sys.argv[1]) + ',' + str(sys.argv[2]) + ',' + str(sys.argv[3]) + ',' + str(sys.argv[4]) + ',' + str(
                sys.argv[5]) + ',' + str(sys.argv[6]) + ',' + str(sys.argv[7]))
        hs.write("\n")
    for testcase in TestCase:
        if testcase == TestCase.training:
            runterm = None
            riscv, file, file_preprocessed, model_name = get_file_names(processor, program,
                                                                        testcase, runterm, memory_port, model_option,
                                                                        feature_size)
            print("### file is being preprocessing ###")
            training_preprocessing(file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo, interval_max)
            accuracy, model = training_offline(file_preprocessed, feature_size, interval_max,
                                               test_percentage, model_option, learning_rate_offline, model_name)
        else:
            for runterm in run:
                riscv, file, file_preprocessed, model_name = get_file_names(processor, program,
                                                                            testcase, runterm, memory_port,
                                                                            model_option, feature_size)
                print("### file is being preprocessing ###")
                training_preprocessing(file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo,
                                       interval_max)
                accuracy = evaluation_offline(model_name, model_option, file_preprocessed, interval_max, feature_size)
                average_acc = average_acc + accuracy
            average_acc = average_acc / 3  # 3 should change with the number of evaluation datasets
            print("### average_acc = ", average_acc)
            with open("accuracy_evaluation.txt", "a") as hs:
                hs.write("evaluation accuracy: " + str(average_acc))
                hs.write("\n")
                hs.write("--------------------")
                hs.write("\n")

elif testing:
    testcase = TestCase.test
    average_models = []
    var_models = []
    for models in chosen_models:
        average_acc = 0
        accuracy_arr = np.array([])
        for runterm in run:
            riscv, file, file_preprocessed, model_name = get_file_names(processor, program,
                                                                        testcase, runterm, memory_port, model_option,
                                                                        feature_size, testing, models)
            print("### file is being preprocessing ###")
            if not os.path.isfile(file_preprocessed):
                training_preprocessing(file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo,
                                       interval_max)
            else:
                print("file exist:", file_preprocessed)
            accuracy = evaluation_offline(model_name, model_option, file_preprocessed, interval_max, feature_size)
            accuracy_arr = np.append(accuracy_arr, accuracy)
            average_acc = average_acc + accuracy
        average_acc = average_acc / 10.0
        variance = np.var(accuracy_arr)
        average_models.append(average_acc)
        var_models.append(variance)
        print(f'### average_acc for {models}= {average_acc}.')
        print(f'### variance_acc for {models}= {variance}.')
    print("average acc for models = ", average_models)
    print("variance acc for models = ", var_models)
    os.remove(file_preprocessed + ".npz")


elif flt16:
    testcase = TestCase.test
    average_acc = 0
    for runterm in run:
        riscv, file, file_preprocessed, model_name = get_file_names(processor, program,
                                                                    testcase, runterm, memory_port, model_option,
                                                                    feature_size)
        print("### file is being preprocessing ###")
        training_preprocessing(file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo, interval_max)
        accuracy = evaluation_offline(model_name, model_option, file_preprocessed, interval_max, feature_size)
        average_acc = average_acc + accuracy
    average_acc = average_acc / 3
    print("### average_acc before optimization= ", average_acc)
    model = models.load_model(model_name)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_saved_model(model_name)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_new_converter = True
    # converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    tflite_models_dir = pathlib.Path("result/model/tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir / "tmp_model.tflite"
    tflite_model_file.write_bytes(tflite_model)

    # set the optimizations flag to use default optimizations

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_fp16_model = converter.convert()
    # set store path
    tflite_model_fp16_file = tflite_models_dir / "tmp_model_quant_f16.tflite"
    tflite_model_fp16_file.write_bytes(tflite_fp16_model)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()

    interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
    interpreter_fp16.allocate_tensors()
    average_acc = 0

    for runterm in run:
        riscv, file, file_preprocessed, model_name = get_file_names(processor, program,
                                                                    testcase, runterm, memory_port, model_option,
                                                                    feature_size)
        print("### file is being preprocessing ###")
        training_preprocessing(file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo, interval_max)
        accuracy = evaluation_flt16(interpreter_fp16, model_option, file_preprocessed, interval_max, feature_size)
        print(accuracy)
        average_acc = average_acc + accuracy
    average_acc = average_acc / 3
    print("### average_acc float16 quantization = ", average_acc)

elif int8:
    testcase = TestCase.test
    average_acc = 0
    for runterm in run:
        riscv, file, file_preprocessed, model_name = get_file_names(processor, program,
                                                                    testcase, runterm, memory_port, model_option,
                                                                    feature_size)
        print("### file is being preprocessing ###")
        training_preprocessing(file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo, interval_max)
        accuracy = evaluation_offline(model_name, model_option, file_preprocessed, interval_max, feature_size)
        average_acc = average_acc + accuracy
    average_acc = average_acc / 3
    print("### average_acc before optimization= ", average_acc)

    model = models.load_model(model_name)
    model.summary()

    if model_option == "transformer_model":
        converter = tf.lite.TFLiteConverter.from_saved_model(model_name)


        def representative_data_gen():
            for input_value in tf.data.Dataset.from_tensor_slices(input_data).batch(1).take(5000):
                yield [input_value[:, feature_size[1]:], input_value[:, 0:ENCODE_DIM],
                       input_value[:, ENCODE_DIM:feature_size[1]]]
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)


        def representative_data_gen():
            for input_value in tf.data.Dataset.from_tensor_slices(input_data).batch(1).take(5000):
                yield [input_value]

    testcase = TestCase.training
    runterm = None
    riscv, file, file_preprocessed, model_name = get_file_names(processor, program,
                                                                testcase, runterm, memory_port, model_option,
                                                                feature_size)
    training_preprocessing(file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo, interval_max)
    input_data = np.load(file_preprocessed + ".npz")['features']
    if model_option == "LSTM" or model_option == "cnn":
        input_data = np.expand_dims(input_data, axis=2).astype(np.float32)

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_new_converter = True
    # converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    tflite_models_dir = pathlib.Path("result/model/tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir / "tmp_model.tflite"
    tflite_model_file.write_bytes(tflite_model)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("input details =  ", input_details)
    print("output details =  ", output_details)

    print("has changed to TFlite")
    # set the optimizations flag to use default optimizations

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8_model = converter.convert()
    # set store path
    tflite_model_int8_file = tflite_models_dir / "tmp_model_quant_int8.tflite"
    tflite_model_int8_file.write_bytes(tflite_int8_model)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()

    interpreter_int8 = tf.lite.Interpreter(model_path=str(tflite_model_int8_file))
    input_type = interpreter_int8.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter_int8.get_output_details()[0]['dtype']
    print('output: ', output_type)
    interpreter_int8.allocate_tensors()
    average_acc = 0

    for runterm in run:
        riscv, file, file_preprocessed, model_name = get_file_names(processor, program,
                                                                    testcase, runterm, memory_port, model_option,
                                                                    feature_size)
        print("### file is being preprocessing ###")
        training_preprocessing(file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo, interval_max)
        accuracy = evaluation_int8(interpreter_int8, model_option, file_preprocessed, interval_max, feature_size)
        print(accuracy)
        average_acc = average_acc + accuracy
    average_acc = average_acc / 3
    print("### average_acc integer8 quantization = ", average_acc)


elif prune == True:
    testcase = TestCase.test
    average_acc = 0
    for runterm in run:
        riscv, file, file_preprocessed, model_name = get_file_names(processor, program,
                                                                    testcase, runterm, memory_port, model_option,
                                                                    feature_size)
        print("### file is being preprocessing ###")
        training_preprocessing(file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo, interval_max)
        accuracy = evaluation_offline(model_name, model_option, file_preprocessed, interval_max, feature_size)
        average_acc = average_acc + accuracy
    average_acc = average_acc / 3
    print("### average_acc before pruning= ", average_acc)

    testcase = TestCase.training
    runterm = None
    riscv, file, file_preprocessed, model_name = get_file_names(processor, program,
                                                                testcase, runterm, memory_port, model_option,
                                                                feature_size)
    print("### file is being preprocessing ###")
    training_preprocessing(file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo, interval_max)
    pruning_training(file_preprocessed, feature_size, interval_max, test_percentage, model_option,
                     learning_rate_offline, model_name)
