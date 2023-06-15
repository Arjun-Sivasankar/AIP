from training_helper import *
from training_preprocessing import *
import os
import numpy as np
import pandas as pd
from keras import models, losses
from keras.layers import Dense, Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import time
from pathlib import Path


gpus = tf.config.list_physical_devices('GPU')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print("Num GPUs Available: ", len(gpus))
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)

#parameter 
iteration = False
dataset = False
config = False

plot = False
jump = False
testing = False

feature_size = [0, 10]
step = 1
test_percentage = 0.33
hash_shift = 2
hash_modulo = 5
dummy_data = False
interval_max = 32
learning_rate_offline = 1e-3
#file name

processor = Processor.riscv
program = Program.basicmath
memory_port = MemoryPort.dram
model_option = Model.cnn

class Objective(object):
    def __init__(self, X_train, X_test, y_train, y_test, X_run, y_run, dir_save,
                 max_epochs, early_stop, learn_rate_epochs,
                 feature_size, interval_max):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_run = X_run
        self.y_run = y_run
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.dir_save = dir_save
        self.learn_rate_epochs = learn_rate_epochs
        self.feature_size = feature_size
        self.interval_max = interval_max
 
    def __call__(self, trial):        
        num_cnn_blocks = trial.suggest_int('num_cnn_blocks', 0, 4)
        num_filters = trial.suggest_categorical('num_filters', [16, 32, 64])
        kernel_size = trial.suggest_int('kernel_size', 2, 3)
        batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096])
                          
        dict_params = {'num_cnn_blocks':num_cnn_blocks,
                       'num_filters':num_filters,
                       'kernel_size':kernel_size,
                       'batch_size':batch_size,
                       }
                                              
        # start of cnn coding 
        model = models.Sequential()  
        if feature_size[0] == feature_size[1]:
            model.add(Input(shape=(feature_size[1],2)))
        else:
            model.add(Input(shape=(feature_size[0] + feature_size[1],1)))
         
        # 1st cnn block

        # additional cnn blocks
        for iblock in range(dict_params['num_cnn_blocks'] - 1):
            model.add(Conv1D(filters=dict_params['num_filters'],
                       kernel_size=dict_params['kernel_size'],
                       activation='relu',padding='same'))
            #model.add(MaxPooling1D(pool_size=2))
        #x = BatchNormalization()(input_tensor)
        #x = Activation('relu')(x)
        model.add(Conv1D(filters=dict_params['num_filters'],
                   kernel_size=dict_params['kernel_size'],
                   activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
         

        # mlp
        model.add(Flatten())
        model.add(Dense(self.interval_max, activation='softmax'))
        model.summary()
        # instantiate and compile model

        opt = Adam()  # default = 0.001
        loss_fn = losses.SparseCategoricalCrossentropy()
        model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])
         
         
        # callbacks for early stopping and for learning rate reducer
        fn = self.dir_save + str(trial.number) + '_cnn.h5'
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=self.early_stop),                     
                          ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                            patience=self.learn_rate_epochs, 
                                            verbose=2, mode='auto', min_lr=1.0e-6),
                          ModelCheckpoint(filepath=fn,
                                          monitor='val_loss', save_best_only=True)]
             
        # fit the model
        print(X_train.shape)
        print(y_train.shape)
        h = model.fit(x=self.X_train, y=self.y_train,
                          batch_size=dict_params['batch_size'],
                          epochs=self.max_epochs,
                          validation_split=0.2,
                          shuffle=True, verbose=2,
                          callbacks=callbacks_list)
        validation_acc = 0
        best_model = models.load_model(fn)
        for runterm in run:
            loss, accuracy = best_model.evaluate(X_run[runterm], y_run[runterm], batch_size=512, verbose=2)
            validation_acc = accuracy + validation_acc
        validation_acc = validation_acc / 3 
                 
        return validation_acc



if __name__ == '__main__':    
    X_run = {}
    y_run = {}
    maximum_epochs = 100
    early_stop_epochs = 5
    learning_rate_epochs = 5
    optimizer_direction = 'maximize'
    number_of_random_points = 25  # random searches to start opt process
    maximum_time = 4*60*60  # seconds

    # fixed parameters - production
    threshold_error = 0.04  # validation loss
    number_of_models = 25

    for testcase in TestCase:
        if testcase == TestCase.training:
            runterm = None
            riscv, file, file_preprocessed, model_name = get_file_names(processor, program, 
                                              testcase,runterm, memory_port, model_option, feature_size)
            file_preprocessed_training = file_preprocessed
            training_preprocessing (file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo)
            input_data = np.load(file_preprocessed+".npz")['features']
            label = np.load(file_preprocessed+".npz")['label']
            label = np.vectorize(min)(label, interval_max - 1)
            X_train, X_test, y_train, y_test = data_split(input_data, label, test_percentage, 43 )
        else:
            for runterm in run:
                riscv, file, file_preprocessed, model_name = get_file_names(processor, program, 
                                              testcase, runterm, memory_port, model_option, feature_size)
                training_preprocessing (file, file_preprocessed, riscv, feature_size, hash_shift, hash_modulo)
                input_data = np.load(file_preprocessed+".npz")['features']
                label = np.load(file_preprocessed+".npz")['label']
                X_run[runterm] = input_data
                y_run[runterm] = label
    
    start_time_total = time.time()
                 
    results_directory = "result/optimization/" 
    if not Path(results_directory).is_dir():
        os.mkdir(results_directory)
                 
    objective = Objective(X_train, X_test, y_train, y_test, X_run, y_run, results_directory,
                            maximum_epochs, early_stop_epochs,
                            learning_rate_epochs, feature_size, interval_max)
     
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=optimizer_direction,
            sampler=TPESampler(n_startup_trials=number_of_random_points))
         
    study.optimize(objective, timeout=maximum_time)
         
    # save results
    df_results = study.trials_dataframe()
    df_results.to_pickle(results_directory + str(processor)+ "/" + str(program) + "/" + str(model_option)  + str(feature_size[0]) + "_" + str(feature_size[1]) + 'df_optuna_results.pkl')
    df_results.to_csv(results_directory + str(processor)+ "/" + str(program) + "/" + str(model_option)  + str(feature_size[0]) + "_" + str(feature_size[1]) + 'df_optuna_results.csv')
         
         
    elapsed_time_total = (time.time()-start_time_total)/60
    print('\n\ntotal elapsed time =',elapsed_time_total,' minutes')

        

                