from training_helper import *
from tensorflow import keras
from keras import models
from model import *
import random
import os
import pathlib
# import tensorflow_model_optimization as tfmot

ENCODE_DIM = 20

def training_offline(file_preprocessed, feature_size, interval_max, test_percentage, model_option, learning_rate, model_name):
    input_data = np.load(file_preprocessed+".npz")['features']
    label = np.load(file_preprocessed+".npz")['label']
    
    batch_size = 4096
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    print(input_data.shape)
    print(label.shape)

    callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath= model_name, monitor='val_loss'
                                       , save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                         patience=5, 
                                         verbose=2, mode='auto', min_lr=1.0e-6)
    ]


    #rand = random.randint(1, 10000)
    X_train, X_test, y_train, y_test = data_split(input_data, label, test_percentage, 43 )
    if model_option == "dense":
        model = dense_model(feature_size, interval_max)
    elif model_option == "cnn":
        model = cnn_model(feature_size, interval_max)
    elif model_option == "LSTM" :
        model = LSTM_model(feature_size, interval_max)
    elif model_option == "GRU" :
        model = GRU_model(feature_size, interval_max)
    elif model_option == "attention_LSTM_model":
        model = attention_LSTM(feature_size, interval_max)
    elif model_option == "attention_dense_model":
        model = attention_dense(feature_size, interval_max)
    elif model_option == "transformer_encoder":
        model = transformer_encoder(feature_size, interval_max)
    elif model_option == "transformer_model":
        model = transformer(feature_size, interval_max)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    if model_option == "transformer_model" or model_option == "attention_LSTM_model":
        DECODE_DIM = feature_size[1]-ENCODE_DIM
        if feature_size [0] ==1 or feature_size[2] == 1:
            model.fit([X_train[:, 0:ENCODE_DIM],X_train[:,ENCODE_DIM:feature_size[1]],X_train[:,feature_size[1]:]], y_train, epochs=50, 
                      validation_split=0.2, batch_size= batch_size, verbose=2 ,callbacks=callbacks)
            
            loss, accuracy = model.evaluate([X_test[:,0:ENCODE_DIM], X_test[:,ENCODE_DIM:feature_size[1]],X_test[:,feature_size[1]:]], y_test, 
                                            batch_size=batch_size, verbose=2)
        else:
            loss, accuracy = model.evaluate([X_test[:,0:ENCODE_DIM ], X_test[:,ENCODE_DIM: ]], y_test, batch_size=batch_size, verbose=2)
    else:
        model.fit(X_train, y_train, epochs=60, validation_split=0.2, batch_size= batch_size, verbose=2 , callbacks=callbacks)
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)

    print("training accuracy = ", accuracy)
    os.remove(file_preprocessed+".npz")
    return accuracy, model


def evaluation_offline(model_name, model_option, file_preprocessed, interval_max, feature_size ):
    
    input_data = np.load(file_preprocessed+".npz")['features']
    label = np.load(file_preprocessed+".npz")['label']
    label = np.vectorize(min)(label, interval_max - 1)
    batch_size = 512
    model = models.load_model(model_name)
    model.summary()
    if model_option == "transformer_model" or model_option == "attention_LSTM_model":
        DECODE_DIM = feature_size[1]-ENCODE_DIM
        if feature_size [0] ==1 or feature_size[2] == 1:
            #split preprocessing data into encoder input, decoder input and dense layer input in Transformer
            loss, accuracy = model.evaluate([input_data[:,0:ENCODE_DIM], input_data[:,ENCODE_DIM:feature_size[1]],input_data[:,feature_size[1]:]], label, 
                                            batch_size=batch_size, verbose=2)
        else:
            loss, accuracy = model.evaluate([input_data[:,0:ENCODE_DIM], input_data[:,ENCODE_DIM:]], label, batch_size=batch_size, verbose=2)
    else:
        loss, accuracy = model.evaluate(input_data, label, batch_size=batch_size, verbose=2)

    print("evaluation_accuracy =  ", accuracy)
    return accuracy


def evaluation_flt16 (interpreter , model_option, file_preprocessed, interval_max, feature_size ):
   
    input_data = np.load(file_preprocessed+".npz")['features']
    label = np.load(file_preprocessed+".npz")['label']
    label = np.vectorize(min)(label, interval_max - 1)
    #Pre-processing: add batch dimension for LSTM
    if model_option == "LSTM" or model_option == "cnn":
        input_data = np.expand_dims(input_data, axis=2).astype(np.float32)
    else:
        input_data = input_data.astype(np.float32)
    print(input_data.shape)
    #resize the input shape to match the batch size
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("input details =  " , input_details)
    print("output details =  " , output_details)

    if model_option == "transformer_model" :
       if feature_size [0] ==1 or feature_size[2] == 1:
           DECODE_DIM = feature_size[1]-ENCODE_DIM
           interpreter.resize_tensor_input(input_details[0]['index'],[512,feature_size [0]+feature_size[2]])
           interpreter.resize_tensor_input(input_details[1]['index'],[512,ENCODE_DIM])
           interpreter.resize_tensor_input(input_details[2]['index'],[512,DECODE_DIM])
       else:
           interpreter.resize_tensor_input(input_details[0]['index'],[512,ENCODE_DIM])
           interpreter.resize_tensor_input(input_details[1]['index'],[512,DECODE_DIM])
    elif model_option ==  "dense" or model_option ==  "transformer_encoder":
        interpreter.resize_tensor_input(input_details[0]['index'],[512,feature_size [0]+feature_size[2]+ feature_size[1]])    
    else:
       interpreter.resize_tensor_input(input_details[0]['index'],[512,feature_size [0]+feature_size[2]+ feature_size[1],1])
   
    interpreter.allocate_tensors()
    output_index = output_details[0]["index"]
    input_details = interpreter.get_input_details()
    print("input details =  " , input_details)
    print("output details =  " , output_details)

    # Run predictions on every  in the "test" dataset.
    prediction_labels = np.array([],dtype=int)
    accurate_count = 0

    for i in range(len(input_data)//512):
        #set input tensor accroding to batch size 512 (can be changed to other bathch size)
        
        if model_option == "transformer_model" :
            if feature_size [0] ==1 or feature_size[2] == 1:
                interpreter.set_tensor(input_details[0]['index'], input_data[i*512:(i+1)*512,feature_size[1]: ])
                interpreter.set_tensor(input_details[1]['index'], input_data[i*512:(i+1)*512,0:ENCODE_DIM])
                interpreter.set_tensor(input_details[2]['index'], input_data[i*512:(i+1)*512,ENCODE_DIM:feature_size[1]])
            else:
                interpreter.set_tensor(input_details[0]['index'], input_data[i*512:(i+1)*512,0:ENCODE_DIM])
                interpreter.set_tensor(input_details[1]['index'], input_data[i*512:(i+1)*512,ENCODE_DIM:feature_size[1]])
        elif model_option ==  "dense" or model_option ==  "transformer_encoder":
            interpreter.set_tensor(input_details[0]['index'], input_data[i*512:(i+1)*512, :])
        else:
            interpreter.set_tensor(input_details[0]['index'], input_data[i*512:(i+1)*512, :, :])

        # Run inference.
        interpreter.invoke()
        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        predictions = interpreter.get_tensor(output_details[0]['index'])
        prediction_label = np.argmax(predictions, axis=1)
        prediction_labels = np.append(prediction_labels, prediction_label)
        interpreter.reset_all_variables()
    
    #infer the remaining data
    remain = len(input_data) % 512

    if model_option == "transformer_model" :
        if feature_size [0] ==1 or feature_size[2] == 1:
            interpreter.resize_tensor_input(input_details[0]['index'],[remain,feature_size [0]+feature_size[2]])
            interpreter.resize_tensor_input(input_details[1]['index'],[remain,ENCODE_DIM])
            interpreter.resize_tensor_input(input_details[2]['index'],[remain,DECODE_DIM])
        else:
            interpreter.resize_tensor_input(input_details[0]['index'],[remain,ENCODE_DIM])
            interpreter.resize_tensor_input(input_details[1]['index'],[remain,DECODE_DIM])
    elif model_option ==  "dense" or model_option ==  "transformer_encoder":
        interpreter.resize_tensor_input(input_details[0]['index'],[remain,feature_size [0]+feature_size[2]+ feature_size[1]]) 
    else:
        interpreter.resize_tensor_input(input_details[0]['index'],[remain,feature_size [0]+feature_size[2]+ feature_size[1],1])
    interpreter.allocate_tensors()


    if model_option == "transformer_model" :
        if feature_size [0] ==1 or feature_size[2] == 1:
            interpreter.set_tensor(input_details[0]['index'], input_data[(len(input_data)//512)*512:,feature_size[1]: ])
            interpreter.set_tensor(input_details[1]['index'], input_data[(len(input_data)//512)*512:,0:ENCODE_DIM])
            interpreter.set_tensor(input_details[2]['index'], input_data[(len(input_data)//512)*512:,ENCODE_DIM:feature_size[1]])
        else:
            interpreter.set_tensor(input_details[0]['index'], input_data[(len(input_data)//512)*512:,0:ENCODE_DIM])
            interpreter.set_tensor(input_details[1]['index'], input_data[(len(input_data)//512)*512:,ENCODE_DIM:feature_size[1]])
    elif model_option ==  "dense" or model_option ==  "transformer_encoder":
            interpreter.set_tensor(input_details[0]['index'], input_data[(len(input_data)//512)*512:, :])
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data[(len(input_data)//512)*512:, :, :])

    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    prediction_label = np.argmax(predictions, axis=1)
    prediction_labels = np.append(prediction_labels, prediction_label)

    # Compare prediction results with ground truth labels to calculate accuracy.
    for index in range(len(prediction_labels)):
        if prediction_labels[index] == label[index]:
           accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_labels)
    print(accuracy)
    return accuracy


def evaluation_int8 (interpreter , model_option, file_preprocessed, interval_max, feature_size ):
   
    input_data = np.load(file_preprocessed+".npz")['features']
    label = np.load(file_preprocessed+".npz")['label']
    label = np.vectorize(min)(label, interval_max - 1)

    label = label.astype(np.int16)
    print(input_data.dtype)
    #Pre-processing: add batch dimension for LSTM
    if model_option == "LSTM" or model_option == "cnn":
        input_data = np.expand_dims(input_data, axis=2).astype(np.float32)
    else:
        input_data = input_data.astype(np.float32)
   
    print(input_data.shape)
    #resize the input shape to match the batch size
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("input details =  " , input_details)
    print("output details =  " , output_details)

    if model_option == "transformer_model" :
       if feature_size [0] ==1 or feature_size[2] == 1:
           DECODE_DIM = feature_size[1]-ENCODE_DIM
           interpreter.resize_tensor_input(input_details[0]['index'],[512,feature_size [0]+feature_size[2]])
           interpreter.resize_tensor_input(input_details[1]['index'],[512,ENCODE_DIM])
           interpreter.resize_tensor_input(input_details[2]['index'],[512,DECODE_DIM])
       else:
           interpreter.resize_tensor_input(input_details[0]['index'],[512,ENCODE_DIM])
           interpreter.resize_tensor_input(input_details[1]['index'],[512,DECODE_DIM])
    elif model_option ==  "dense" or model_option ==  "transformer_encoder":
        interpreter.resize_tensor_input(input_details[0]['index'],[512,feature_size [0]+feature_size[2]+ feature_size[1]]) 
    else:
       interpreter.resize_tensor_input(input_details[0]['index'],[512,feature_size [0]+feature_size[2]+ feature_size[1],1])
   
    interpreter.allocate_tensors()
    output_index = output_details[0]["index"]
    input_details = interpreter.get_input_details()
    print("input details =  " , input_details)
    print("output details =  " , output_details)

    # Run predictions on every  in the "test" dataset.
    prediction_labels = np.array([],dtype=int)
    accurate_count = 0

    for i in range(len(input_data)//512):
    #set input tensor accroding to batch size
        if model_option == "transformer_model" :
            if feature_size [0] ==1 or feature_size[2] == 1:
                interpreter.set_tensor(input_details[0]['index'], input_data[i*512:(i+1)*512,feature_size[1]: ])
                interpreter.set_tensor(input_details[1]['index'], input_data[i*512:(i+1)*512,0:ENCODE_DIM])
                interpreter.set_tensor(input_details[2]['index'], input_data[i*512:(i+1)*512,ENCODE_DIM:feature_size[1]])
            else:
                interpreter.set_tensor(input_details[0]['index'], input_data[i*512:(i+1)*512,0:ENCODE_DIM])
                interpreter.set_tensor(input_details[1]['index'], input_data[i*512:(i+1)*512,ENCODE_DIM:feature_size[1]])
        elif model_option ==  "dense" or model_option ==  "transformer_encoder":
            interpreter.set_tensor(input_details[0]['index'], input_data[i*512:(i+1)*512, :])
        else:
            interpreter.set_tensor(input_details[0]['index'], input_data[i*512:(i+1)*512, :, :])

        # Run inference.
        interpreter.invoke()
        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        predictions = interpreter.get_tensor(output_details[0]['index'])
        prediction_label = np.argmax(predictions, axis=1)
        prediction_labels = np.append(prediction_labels, prediction_label)
        interpreter.reset_all_variables()
    
    #infer the remaining data
    remain = len(input_data) % 512

    if model_option == "transformer_model" :
        if feature_size [0] ==1 or feature_size[2] == 1:
            interpreter.resize_tensor_input(input_details[0]['index'],[remain,feature_size [0]+feature_size[2]])
            interpreter.resize_tensor_input(input_details[1]['index'],[remain,ENCODE_DIM])
            interpreter.resize_tensor_input(input_details[2]['index'],[remain,DECODE_DIM])
        else:
            interpreter.resize_tensor_input(input_details[0]['index'],[remain,ENCODE_DIM])
            interpreter.resize_tensor_input(input_details[1]['index'],[remain,DECODE_DIM])
    elif model_option ==  "dense" or model_option ==  "transformer_encoder":
        interpreter.resize_tensor_input(input_details[0]['index'],[remain,feature_size [0]+feature_size[2]+ feature_size[1]])
    else:
        interpreter.resize_tensor_input(input_details[0]['index'],[remain,feature_size [0]+feature_size[2]+ feature_size[1],1])
    interpreter.allocate_tensors()


    if model_option == "transformer_model" :
        if feature_size [0] ==1 or feature_size[2] == 1:
            interpreter.set_tensor(input_details[0]['index'], input_data[(len(input_data)//512)*512:,feature_size[1]: ])
            interpreter.set_tensor(input_details[1]['index'], input_data[(len(input_data)//512)*512:,0:ENCODE_DIM])
            interpreter.set_tensor(input_details[2]['index'], input_data[(len(input_data)//512)*512:,ENCODE_DIM:feature_size[1]])
        else:
            interpreter.set_tensor(input_details[0]['index'], input_data[(len(input_data)//512)*512:,0:ENCODE_DIM])
            interpreter.set_tensor(input_details[1]['index'], input_data[(len(input_data)//512)*512:,ENCODE_DIM:feature_size[1]])
    elif model_option ==  "dense" or model_option ==  "transformer_encoder":
            interpreter.set_tensor(input_details[0]['index'], input_data[(len(input_data)//512)*512:, :])
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data[(len(input_data)//512)*512:, :, :])

    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    prediction_label = np.argmax(predictions, axis=1)
    prediction_labels = np.append(prediction_labels, prediction_label)

    # Compare prediction results with ground truth labels to calculate accuracy.
    print(len(label))
    for index in range(len(prediction_labels)):
        if prediction_labels[index] == label[index]:
           accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_labels)
    print(accuracy)
    interpreter.reset_all_variables()
    return accuracy

def pruning_training(file_preprocessed, feature_size, interval_max, test_percentage, model_option, learning_rate, model_name):
    batch_size = 4096
    epochs = 10
    
    input_data = np.load(file_preprocessed+".npz")['features']
    label = np.load(file_preprocessed+".npz")['label']

    X_train, X_test, y_train, y_test = data_split(input_data, label, test_percentage, 43 )
    model = models.load_model(model_name)
    model.summary()

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    
    validation_split = 0.2 # 20% of training set will be used for validation set. 

    num_data = X_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_data / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=0.80,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    model_for_pruning.summary()

    logdir = "result/model/pruning"

    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(X_train, y_train,
                    batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                    callbacks=callbacks, verbose=0)
    
    model_for_pruning_accuracy = model_for_pruning.evaluate(
    X_test, y_test, verbose=0)

    print("pruning acc = ", model_for_pruning_accuracy )





    








