#simulate online training 

from training_helper import *
from model import *
import time


def training_online(hash_shift, hash_modulo, feature_size, file, model_option ,model_name, load_model, riscv, interval_max):
    # parameter
    max_rows = int(1e6)

    training_batch = 100
    accuracy_log_interval = 100
    epochs = 1
    validation_split = 0.05
    learning_rate = 1e-3

    address_ld, interval_ld = data_import(file, max_rows=max_rows, riscv=riscv)

    # set maximum interval length
    interval_mod = np.zeros(len(interval_ld))
    interval_mod = np.vectorize(min)(interval_ld, interval_max - 1)
    print("### intervals preprocessed ###")

    # preprocess input data
    features, label = data_preprocessing(address_ld, interval_mod, hash_shift, hash_modulo, feature_size)

    # instantiate an optimizer to train the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # instantiate a loss function.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # prepare the metrics
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    train_epoch_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # define model
    if load_model == False:
        if model_option == "dense_model":
            model = dense_model(feature_size, interval_max)
        elif model_option == "LSTM_model" :
            model = LSTM_model(feature_size, interval_max)
        elif model_option == "GRU_model" :
            model = GRU_model(feature_size, interval_max)
    else:
        model = keras.models.load_model(model_name)
        print(f'#####load_model: {model_name} ####')
    model.compile(optimizer = optimizer, loss = loss_fn , metrics = train_acc_metric)
    feature_num = feature_size[0] + feature_size[1]
    # prepare the training dataset
    x_train = features
    y_train = label
    x_train = np.reshape(x_train, (-1, feature_num))



    # reserve samples for validation
    val_size = int(len(interval_mod) * validation_split)
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    print(x_train.shape)
    print(y_train.shape)
    # prepare the training dataset (not shuffled for quasi online learning)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(training_batch)
    # prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(training_batch)


    # @tf.function
    def train_step(x, y, step):
        with tf.GradientTape() as tape:
            logits2 = model(x, training=False)
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        # update training metric
        train_acc_metric.update_state(y, logits2)
        train_epoch_acc_metric.update_state(y, logits2)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
        # log every several batches
        if step % accuracy_log_interval == 0:
            print("training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
            print("seen so far: %d samples" % ((step + 1) * training_batch))

            # display metrics at the end of each mini-batch
            train_acc = train_acc_metric.result()
            print("training acc over mini-batch: %.4f" % (float(train_acc),))

        # reset training metrics at the end of mini-batch
        train_acc_metric.reset_states()
        return loss_value


    # @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)


    for epoch in range(epochs):
        print("\nstart of epoch %d" % (epoch,))
        start_time = time.time()

        # iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train, step)

        # display metrics at the end of each epoch
        train_epoch_acc = train_epoch_acc_metric.result()
        print("training acc over epoch: %.4f" % (float(train_epoch_acc),))

        # reset training metrics at the end of each epoch
        train_epoch_acc_metric.reset_states()

        # run a validation loop at the end of each epoch
        for x_batch_val, y_batch_val in val_dataset:
            test_step(x_batch_val, y_batch_val)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("validation acc: %.4f" % (float(val_acc),))
        print("time taken: %.2fs" % (time.time() - start_time))

