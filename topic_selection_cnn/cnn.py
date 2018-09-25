from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
# from plot_metrics import plot_accuracy, plot_loss, plot_roc_curve

from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, \
                         Dropout, MaxPooling1D, GlobalAveragePooling1D, \
                         GlobalMaxPooling1D, Lambda, Concatenate, Dense, regularizers
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers, activations
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

np.random.seed(15)  # for reproducibility

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

K.set_image_dim_ordering('tf')

"""
CNN used to classify spectrograms of normal participants (0) or depressed
participants (1).
(# images, # rows, # cols)
(num_samples, time_dimension, mel_bin_dimension)
"""

def retrieve_file(file_name):
    path = '/media/hdd1/genfyp/processed_nosampling/'
    outfile = path + file_name
    X = np.load(outfile)
    X = X['arr_0']
    return X


def preprocess(X, max_len, num_bins):
    """
    Preprocess input Xs to a numpy array where every training sample is zero padded 
    to constant time dimension (max_len) and contains num_bins frequency bins.
    
    Args:
        X: numpy array of numpy arrays (X), each of which is of different time dimension 
           but same mel dimension (usually 128)
        max_len: Length up to which each np.array in X is padded with 0s
        num_bins: Constant mel dimension
    
    Returns:
        X_proc: single numpy array of shape (X.shape[0], max_len, num_bins), which is fed into 1D CNN  
    """
    X_proc = np.zeros([X.shape[0], max_len, num_bins])
    for idx, x in enumerate(X):
        if x.shape[0] < max_len:
            # Pad sequence (only in time dimension) with 0s
            x = np.pad(x, pad_width=((0, max_len - x.shape[0]), (0,0)), mode='constant')
        else:
            # Trim sequence to be within max_len timesteps
            x = x[:max_len, :]
        # Update processed sequences
        X_proc[idx, :, :] = x
    return X_proc


def cnn(X_train, y_train, X_test, y_test,
        max_len, num_bins, 
        nb_classes, class_weight, 
        batch_size, epochs):
    NUM_CONV_LAYERS = 3
    NUM_DENSE_LAYERS = 1
    NUM_FILTERS = 32
    KERNEL_SIZE = 125
    L2_LAMBDA = 0.01
    DROPOUT = 0.2
    POOL_SIZE = 3
    DENSE_SIZE = 64
    
    print("Config:\nNUM_CONV_LAYERS: {}\nNUM_DENSE_LAYERS: {}\nNUM_FILTERS: {}\nKERNEL_SIZE: {}\nL2_LAMBDA: {}\nDROPOUT: {}\nPOOL_SIZE: {}\nDENSE_SIZE: {}"
          .format(NUM_CONV_LAYERS, NUM_DENSE_LAYERS, NUM_FILTERS, KERNEL_SIZE, L2_LAMBDA, DROPOUT, POOL_SIZE, DENSE_SIZE))

    inputs = Input(batch_shape=(None, max_len, num_bins))
    x = inputs
    for layer_count in range(NUM_CONV_LAYERS - 1):
        x = Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE,
                   padding='valid', 
                   strides=1, dilation_rate=1, # 2**layer_count,
                   activation='linear',
                   kernel_regularizer=regularizers.l2(L2_LAMBDA))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(DROPOUT)(x)
        x = MaxPooling1D(pool_size=POOL_SIZE)(x)
    # Final Conv1D layer doesn't undergo MaxPooling1D
    x = Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE,
               padding='valid', 
               strides=1, dilation_rate=2**layer_count,
               activation='linear',
               kernel_regularizer=regularizers.l2(L2_LAMBDA))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(DROPOUT)(x)

    avgpool = GlobalAveragePooling1D()(x)
    maxpool = GlobalMaxPooling1D()(x)
    l2pool = Lambda(lambda a: K.l2_normalize(K.sum(a, axis=1)))(x)
    x = Concatenate()([avgpool, maxpool, l2pool])
    
    for layer_count in range(NUM_DENSE_LAYERS):
        x = Dense(units=DENSE_SIZE, activation='relu', 
                  kernel_regularizer=regularizers.l2(L2_LAMBDA))(x)
        x = Dropout(DROPOUT)(x)

    x = Dense(nb_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    # sgd_nest = optimizers.SGD(lr=0.01, momentum=0.00, decay=0.0, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics=['accuracy'])
    model.summary()
    
    history = model.fit(X_train, y_train, 
                        batch_size=batch_size, epochs=epochs,
                        validation_data=(X_test, y_test),
                        class_weight=class_weight,
                        verbose=1)

    # Evaluate accuracy on test and train sets
    score_train = model.evaluate(X_train, y_train, verbose=0)
    print('Train accuracy:', score_train[1])
    score_test = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score_test[1])

    return model, history


def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    cnf_matrix = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix: (sklearn)\n")
    print(cnf_matrix)
    [[tn, fp], [fn, tp]] = cnf_matrix
    return np.array([[tp, fp], [fn, tn]])


def model_performance(model, X_train, X_test, y_train, y_test):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = np.argmax(model.predict(X_test), axis=-1)
    y_train_pred = np.argmax(model.predict(X_train), axis=-1)

    # Computing confusion matrix for test dataset
    conf_matrix = standard_confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:\n")
    print(conf_matrix)
    
    target_name=['non-depressed','depressed']
    clf_report = classification_report(y_test, y_test_pred, target_names=target_name)
    print("\nClassification Report (sklearn):\n")
    print(clf_report)

#    precision=precision_score(y_test.tolist(), y_test_pred.tolist())
#    print(precision_score(y_test.tolist(), y_test_pred.tolist()))
#    recall=recall_score(y_test.tolist(), y_test_pred.tolist())
#    f1=f1_score(y_test.tolist(), y_test_pred.tolist())  
#    print("=====sklearn=======")
#    print('precision: {0}\nrecall: {1} \nf1: {2}'.format(precision, recall,f1 ))

    return y_train_pred, y_test_pred, conf_matrix, clf_report


if __name__ == "__main__":
    # CNN training parameters
    BATCH_SIZE = 1
    NB_CLASSES = 2
    EPOCHS = 3

    X_train = retrieve_file('train_samples.npz')
    y_train = retrieve_file('train_labels.npz')
    X_test = retrieve_file('test_samples.npz')
    y_test = retrieve_file('test_labels.npz')

    # Maximum time duration among training samples 
    MAX_LEN = np.max([X_train[i].shape[0] for i in range(len(X_train))])
    print("Maximum length of training X: {} (timesteps)".format(MAX_LEN))
    # Number of mel bins in training samples
    NUM_BINS = X_train[0].shape[1]
    print("Number of mel bins: ", NUM_BINS)

    # Preprocess input Xs
    X_train = preprocess(X_train, max_len=MAX_LEN, num_bins=NUM_BINS)
    X_test = preprocess(X_test, max_len=MAX_LEN, num_bins=NUM_BINS)
    print(X_train.shape, X_test.shape)

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
    print(Y_train.shape, Y_test.shape)

    # Compute class weights
    cls_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    print("\nClass weights: ", cls_weight)

    # run CNN
    print('\nFitting model...')
    model, history = cnn(X_train, Y_train, X_test, Y_test, 
                         MAX_LEN, NUM_BINS, 
                         NB_CLASSES, cls_weight,
                         BATCH_SIZE, EPOCHS)

    # evaluate model
    print('\nEvaluating model...')
    y_train_pred, y_test_pred, conf_matrix, clf_report = model_performance(model, X_train, X_test, y_train, y_test)

    # # save model to locally
    # print('Saving model locally...')
    # model_name = '/media/hdd1/genfyp/models/cnn_{}.h5'.format(model_id)
    # model.save(model_name)

    # custom evaluation metrics
    print('\nCalculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}".format(f1_score))