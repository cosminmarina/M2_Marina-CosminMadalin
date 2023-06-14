from __future__ import annotations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score, precision_recall_fscore_support, ConfusionMatrixDisplay
import AutoEncoders
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from typing import Union
import glob
from PIL import Image
import h5py
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import argparse
import warnings
import json

def apply_mask(X: np.ndarray, X_mask: np.ndarray):
    """
    | apply_mask                                        |
    |---------------------------------------------------|
    | Method to apply masks to images.                  |
    |___________________________________________________|
    | ndarray, ndarray, ndarray, ndarray, -> ndarray,   |
    |    ndarray                                        |
    |___________________________________________________|
    | Input:                                            |
    | X: the ndarray of images.                         |
    | X_mask, X_test_mask: the ndarray of masks.        |
    |___________________________________________________|
    | Output:                                           |
    | A ndarray containing the images after apply mask. |
    """
    X = np.resize([X], (np.shape(X) + (1,)))
    return X * X_mask

def load_data(path:str, test_size:float=0.25, get_masks:bool = False, size: tuple=(350,350), verbose:bool = False):
    """
    | load_data                                         |
    |---------------------------------------------------|
    | Function that performs the preprocessing.         |
    |___________________________________________________|
    | str, float, bool -> ndarray, ndarray, ndarray,    |
    |     ndarray                                       |
    |___________________________________________________|
    | Input:                                            |
    | path: where the datasets are placed.              |
    | test_size: the size for train-test split.         |
    | get_masks: flag that indicates to get the masks   |
    |     instead of the images.                        |
    | size: tuple to resize the images.                 |
    | verbose: if info should be printed or no.         |
    |___________________________________________________|
    | Output:                                           |
    | The train and test datasets.                      |
    """
    match_str = '*).png'
    benign_files = glob.glob(path+'benign/'+match_str)
    malignant_files = glob.glob(path+'malignant/'+match_str)
    normal_files = glob.glob(path+'normal/'+match_str)
    if get_masks:
        match_str = '*mask.png'
        benign_files_mask = glob.glob(path+'benign/'+match_str)
        malignant_files_mask = glob.glob(path+'malignant/'+match_str)
        normal_files_mask = glob.glob(path+'normal/'+match_str)

    if verbose:
        print(f"Benign {'mask ' if get_masks else ''}size: {np.shape(benign_files)}")
        print(f"Malignant {'mask ' if get_masks else ''}size: {np.shape(malignant_files)}")
        print(f"Normal {'mask ' if get_masks else ''}size: {np.shape(normal_files)}")

    benign_imgs = np.array([np.array(Image.open((file)).convert('L').resize(size)) for file in benign_files])
    malignant_imgs = np.array([np.array(Image.open((file)).convert('L').resize(size)) for file in malignant_files])
    normal_imgs = np.array([np.array(Image.open((file)).convert('L').resize(size)) for file in normal_files])
    
    if get_masks:
        benign_imgs_mask = np.array([np.array(Image.open((file)).convert('L').resize(size)) for file in benign_files_mask]) / 255
        malignant_imgs_mask = np.array([np.array(Image.open((file)).convert('L').resize(size)) for file in malignant_files_mask]) / 255
        normal_imgs_mask = np.array([np.array(Image.open((file)).convert('L').resize(size)) for file in normal_files_mask]) / 255
        
        benign_imgs_mask = np.resize([(benign_imgs_mask > 0.5).astype(np.int32)], (np.shape(benign_imgs_mask) + (1,)))
        malignant_imgs_mask = np.resize([(malignant_imgs_mask > 0.5).astype(np.int32)], (np.shape(malignant_imgs_mask) + (1,)))
        normal_imgs_mask = np.resize([(normal_imgs_mask > 0.5).astype(np.int32)], (np.shape(normal_imgs_mask) + (1,)))

    y_tags = np.concatenate((np.zeros(np.shape(benign_imgs)[0]), np.ones(np.shape(malignant_imgs)[0]), np.ones(np.shape((normal_imgs))[0])+1))
    X = np.concatenate((benign_imgs, malignant_imgs, normal_imgs),axis=0)
    
    if get_masks:
        X_mask = np.concatenate((benign_imgs_mask, malignant_imgs_mask, normal_imgs_mask),axis=0)
        X = apply_mask(X, X_mask)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_tags, test_size=test_size)

    return X_train, X_test, y_train, y_test

    # print(np.array([np.shape(img) for img in benign_imgs]).min(axis=0))
    # print(np.array([np.shape(img) for img in malignant_imgs]).min(axis=0))
    # print(np.array([np.shape(img) for img in normal_imgs]).min(axis=0))

    # print(np.array([np.shape(img) for img in benign_imgs]).max(axis=0))
    # print(np.array([np.shape(img) for img in malignant_imgs]).max(axis=0))
    # print(np.array([np.shape(img) for img in normal_imgs]).max(axis=0))

def runAE(input_dim: Union[int, list[int]], latent_dim: int, arch: int, with_cpu: bool, n_epochs: int, data_train: Union[np.ndarray, list, tf.data.DataSet], data_test: Union[np.ndarray, list, tf.data.DataSet], file_save: str, verbose: bool):
    """
    | runAE                                             |
    |---------------------------------------------------|
    | Function that performs the AE traing.             |
    |___________________________________________________|
    | int or list of int, int, bool, int, ndarray,      |
    | str, bool -> ndarray, ndarray                     |
    |___________________________________________________|
    | Input:                                            |
    | input_dim: int or list of int that contains the   |
    |    shape of the input data to the keras.model.    |
    | latent_dim: int that represent the shape of the   |
    |    latent (code) space.                           |
    | arch: integer value that determine which model    |
    |    architecture sould be used to build the model. |
    | with_cpu, verbose: booleans values that determines|
    |    if the cpu should be used instead gpu or if the|
    |    information should be displayed, respectively. |
    | n_epochs: int that corresponts to the number of   |
    |    epochs for the keras.model.                    |
    | data_train: data to train the model.              |
    | data_test: test data.                             |
    | file_save: where to save the .h5 model.           |
    |___________________________________________________|
    | Output:                                           |
    | The data encoded.                                 |
    """
    verbose = 1 if verbose else 0
    if with_cpu:
        with tf.device("/cpu:0"):
            AE = AutoEncoders.AE_conv(input_dim=input_dim,latent_dim=latent_dim,arch=arch)

            AE.compile(optimizer='adam', loss='mse')

            history = AE.fit(data_train, data_train, epochs=n_epochs, batch_size=32, min_delta=10, patience=50, verbose=verbose)
    else:
        AE = AutoEncoders.AE_conv(input_dim=input_dim,latent_dim=latent_dim,arch=arch)

        AE.compile(optimizer='adam', loss='mse')

        history = AE.fit(data_train, data_train, epochs=n_epochs, batch_size=32, min_delta=10, patience=50, verbose=verbose)

    plt.figure()
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./history-AE-{file_save[2:-3]}-latent{latent_dim}.png')
    plt.yscale('symlog')
    plt.savefig(f'./history-AE-{file_save[2:-3]}-latent{latent_dim}-logscale.png')
    plt.close()

    AE.encoder.save(file_save)
    return AE.encoder.predict(data_train), AE.encoder.predict(data_test)

def get_stats(y_test: Union[np.ndarray, list, tf.data.DataSet], y_pred: Union[np.ndarray, list, tf.data.DataSet], file_save:str, latent_dim:int=64):
    """
    | get_stats                                         |
    |---------------------------------------------------|
    | Function that obtain the scores of the classifier.|
    |___________________________________________________|
    | ndarray, ndarray, str ->                          |
    |___________________________________________________|
    | Input:                                            |
    | y_test, y_pred: the real and predicted y values.  |
    | file_save: where to save the results.             |
    | latent_dim: int that represent the shape of the   |
    |    latent (code) space.                           |
    |___________________________________________________|
    | Output:                                           |
    | Nothing, we save all in files.                    |
    """
    # RECALL
    rec_gen = recall_score(y_test, y_pred, average='weighted')
    # PRECISION
    prec_gen = precision_score(y_test, y_pred, average='weighted')
    # F1
    f1_gen = f1_score(y_test, y_pred, average='weighted')
    # PER CLASS
    prf = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    gen_stats = np.array([prec_gen, rec_gen, f1_gen])
    df = pd.DataFrame(np.concatenate((prf[:-1], np.array([gen_stats]).T), axis=1), index=['Precision', 'Recall', 'Fscore'], columns=['Benign', 'Malignant', 'Normal', 'General'])
    df.to_csv(f'./stats{file_save[2:-3]}-latent{latent_dim}.csv')

    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(f'./confusion_matrix_clf{file_save[2:-3]}-latent{latent_dim}.png')
    plt.close()


def run_RFC(X_train: Union[np.ndarray, list, tf.data.DataSet], X_test: Union[np.ndarray, list, tf.data.DataSet], y_train: Union[np.ndarray, list, tf.data.DataSet], y_test: Union[np.ndarray, list, tf.data.DataSet], file_save:str, latent_dim:int):
    """
    | run_RFC                                           |
    |---------------------------------------------------|
    | Function that performs the RFC training and       |
    | evaluation.                                       |
    |___________________________________________________|
    | ndarray, ndarray, ndarray, ndarray, str ->        |
    |___________________________________________________|
    | Input:                                            |
    | X_train, X_test, y_train, y_test: the train and   |
    |    test datasets for input X and output y.        |
    | file_save: where to save the results.             |
    | latent_dim: int that represent the shape of the   |
    |    latent (code) space.                           |
    |___________________________________________________|
    | Output:                                           |
    | Nothing, we save all in files.                    |
    """
    clf = RandomForestClassifier()
    parameter_space = {
        'criterion' : ['gini', 'entropy', 'log_loss'],
        'n_estimators' : [50, 100, 200]
    }
    clf = GridSearchCV(clf, parameter_space, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    fig, ax = plt.subplots(1,5, figsize=(10,2))
    for i in range(5):
        tree.plot_tree(clf.best_estimator_.estimators_[i],
                       feature_names=np.arange(latent_dim),
                       class_names=['benign', 'malignant', 'normal'],
                       filled=True,
                       ax = ax[i])
        ax[i].set_title(f'Estimator: {str(i)}', fontsize=10)
    plt.savefig(f'./tree-5estimators-{file_save[2:-3]}-latent{latent_dim}.png')
    plt.close()
    plt.figure()
    tree.plot_tree(clf.best_estimator_.estimators_[i],
                   feature_names=np.arange(latent_dim),
                   class_names=['benign', 'malignant', 'normal'],
                   filled=True)
    plt.savefig(f'./tree-1estimator-{file_save[2:-3]}-latent{latent_dim}.png')
    plt.close()
    data_predicted = clf.predict(X_test)
    get_stats(y_test, data_predicted, file_save, latent_dim)
    
def run_Inception_AUG(X_train: Union[np.ndarray, list, tf.data.DataSet], X_test: Union[np.ndarray, list, tf.data.DataSet], y_train: Union[np.ndarray, list, tf.data.DataSet], y_test: Union[np.ndarray, list, tf.data.DataSet], file_save:str, input_dim: Union[int, list[int]], verbose:bool=False):
    """
    | run_Inception_AUG                                 |
    |---------------------------------------------------|
    | Function that performs the InceptionResNetV2      |
    | training and evaluation wiht AUG.                 |
    |___________________________________________________|
    | ndarray, ndarray, ndarray, ndarray, str, list     |
    |    , bool ->                                      |
    |___________________________________________________|
    | Input:                                            |
    | X_train, X_test, y_train, y_test: the train and   |
    |    test datasets for input X and output y.        |
    | file_save: where to save the results.             |
    | input_dim: int or list of int that contains the   |
    |    shape of the input data to the keras.model.    |
    | verbose: if info should be printed or no.         |
    |___________________________________________________|
    | Output:                                           |
    | Nothing, we save all in files.                    |
    """
    input_tensor = layers.Input(input_dim)
    inceptionresnetv2 = InceptionResNetV2(input_tensor=input_tensor, weights=None, include_top=True, classes=3)
    
    #batch size
    train_batch_size = 16
    val_batch_size = 16
    nepochs = 50
    #model optimizer
    optimizer = Adam(beta_1=0.99,beta_2=0.99,lr=1.0e-4,decay=1.0e-6)
    #loss
    loss= 'categorical_crossentropy'
    #create checkpoint callback to save model on epochs that have best validation accuracy
    checkpoint = ModelCheckpoint('./weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5', monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    inceptionresnetv2.compile(loss=loss, optimizer=optimizer, metrics = ['categorical_accuracy'])  
    
    train_steps_per_epoch = int(np.ceil(len(X_train)/train_batch_size))
    val_steps_per_epoch = int(np.ceil(len(X_train)/val_batch_size))
    
    #Generator for keras model
    seed = 9

    data_gen_args = dict(width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=10,
                        horizontal_flip=True,
                        fill_mode='nearest')

    train_image_datagen = ImageDataGenerator(**data_gen_args)   
    # train_image_datagen = ImageDataGenerator()   
    #val_image_datagen = ImageDataGenerator()   

    training_set_generator = train_image_datagen.flow(X_train, y_train, batch_size=train_batch_size, shuffle=True,seed=seed)
    #validation_set_generator = val_image_datagen.flow(x_val, y_val, batch_size=val_batch_size, shuffle=True, seed=seed)
    
    history = inceptionresnetv2.fit_generator(training_set_generator, steps_per_epoch=train_steps_per_epoch, validation_steps=val_steps_per_epoch, epochs = nepochs, callbacks=callbacks_list, verbose=verbose)
    
    plt.figure()
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./history-inception-{file_save[2:-3]}.png')
    plt.yscale('symlog')
    plt.savefig(f'./history-inception-{file_save[2:-3]}-logscale.png')
    plt.close()
    
    data_predicted = inceptionresnetv2.predict(X_test)
    data_predicted = np.argmax(data_predicted, axis=1)
    y_test = np.argmax(y_test, axis=1)
    get_stats(y_test, data_predicted, file_save[:-3]+'-inception.h5')
    
    
def run_Inception(X_train: Union[np.ndarray, list, tf.data.DataSet], X_test: Union[np.ndarray, list, tf.data.DataSet], y_train: Union[np.ndarray, list, tf.data.DataSet], y_test: Union[np.ndarray, list, tf.data.DataSet], file_save:str, input_dim: Union[int, list[int]], verbose:bool=False):
    """
    | run_Inception                                     |
    |---------------------------------------------------|
    | Function that performs the InceptionResNetV2      |
    | training and evaluation.                          |
    |___________________________________________________|
    | ndarray, ndarray, ndarray, ndarray, str, list     |
    |    , bool ->                                      |
    |___________________________________________________|
    | Input:                                            |
    | X_train, X_test, y_train, y_test: the train and   |
    |    test datasets for input X and output y.        |
    | file_save: where to save the results.             |
    | input_dim: int or list of int that contains the   |
    |    shape of the input data to the keras.model.    |
    | verbose: if info should be printed or no.         |
    |___________________________________________________|
    | Output:                                           |
    | Nothing, we save all in files.                    |
    """
    input_tensor = layers.Input(input_dim)
    inceptionresnetv2 = InceptionResNetV2(input_tensor=input_tensor, weights=None, include_top=True, classes=3)
    
    # categorical
    y_train = keras.utils.to_categorical(y_train, 3)
    y_test = keras.utils.to_categorical(y_test, 3)
    
    # batch size
    train_batch_size = 16
    nepochs = 50
    # model optimizer
    optimizer = Adam(beta_1=0.99,beta_2=0.99,lr=1.0e-4,decay=1.0e-6)
    # loss
    loss = 'categorical_crossentropy'
    
    inceptionresnetv2.compile(loss=loss, optimizer=optimizer, metrics = ['categorical_accuracy'])  
    
    history = inceptionresnetv2.fit(X_train, y_train, batch_size=train_batch_size, verbose=verbose, validation_split=0.25, epochs=nepochs)
    
    plt.figure()
    plt.plot(history.history['loss'],label='loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./history-inception-{file_save[2:-3]}.png')
    plt.close()
    
    data_predicted = inceptionresnetv2.predict(X_test)
    data_predicted = np.argmax(data_predicted, axis=1)
    y_test = np.argmax(y_test, axis=1)
    get_stats(y_test, data_predicted, file_save[:-3]+'-inception.h5')
    
def main():
    file_params_name = 'params.json'
    
    # Parser initialization
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", dest='method', help="Specify an method to execute between: \n 'both' (default), 'ae-rfc', 'inception' or 'latents'")
    parser.add_argument("-f", "--configfile", dest='conf', help="JSON file with configuration of parameters. If not specified and 'method' require the file, it will be searched at 'params.json'")
    args = parser.parse_args()
    
    # Read file
    ## Default if configfile is not specified
    if args.conf is not None:
        file_params_name = args.conf
    else:
        warnings.warn("Not specified configuration file (-f | --configfile), by default it will be searched at 'params.json'")
    ## Try-except-else to open file and re-write params
    try:
        file_params = open(file_params_name)
    except:
        raise OSError(f'File {file_params_name} not found. Your method need a configuration parameters file')
    else:
        params = json.load(file_params)
        file_params.close()
    
    # Default method
    if args.method == 'both' or args.method is None:
        X_train, X_test, y_train, y_test = load_data('./data/Dataset_BUSI_with_GT/', size=tuple(params["input_dim"]), verbose=True)
        if params["verbose"]:
            print('X Train: ', np.shape(X_train))
            print('X Test: ', np.shape(X_test))
            print('y Train: ', np.shape(y_train))
            print('y Test: ', np.shape(y_test))

        X_train_encoded, X_test_encoded = runAE(input_dim=params["input_dim"], latent_dim=params["latent_dim"], arch=params["arch"], with_cpu=params["with_cpu"], n_epochs=params["n_epochs"], data_train=X_train, data_test=X_test, file_save=params["file_save"], verbose=params["verbose"])

        run_RFC(X_train_encoded, X_test_encoded, y_train, y_test, params["file_save"], params["latent_dim"])

        X_train_inception, X_test_inception, y_train_inception, y_test_inception = load_data('./data/Dataset_BUSI_with_GT/', get_masks=True, size=tuple(params["input_dim"]), verbose=True)

        if params["verbose"]:
            print('X Train inception: ', np.shape(X_train_inception))
            print('X Test inception: ', np.shape(X_test_inception))
            print('y Train inception: ', np.shape(y_train_inception))
            print('y Test inception: ', np.shape(y_test_inception))

        run_Inception(X_train_inception, X_test_inception, y_train_inception, y_test_inception, file_save=params["file_save"], input_dim=tuple(params["input_dim"])+(1,), verbose=params["verbose"])
    
    # Only AE-RFC method
    elif args.method == 'ae-rfc':
        X_train, X_test, y_train, y_test = load_data('./data/Dataset_BUSI_with_GT/', size=tuple(params["input_dim"]), verbose=True)
        if params["verbose"]:
            print('X Train: ', np.shape(X_train))
            print('X Test: ', np.shape(X_test))
            print('y Train: ', np.shape(y_train))
            print('y Test: ', np.shape(y_test))

        X_train_encoded, X_test_encoded = runAE(input_dim=params["input_dim"], latent_dim=params["latent_dim"], arch=params["arch"], with_cpu=params["with_cpu"], n_epochs=params["n_epochs"], data_train=X_train, data_test=X_test, file_save=params["file_save"], verbose=params["verbose"])

        run_RFC(X_train_encoded, X_test_encoded, y_train, y_test, params["file_save"], params["latent_dim"])
    
    # Only Inception method
    elif args.method == 'inception':
        X_train, X_test, y_train, y_test = load_data('./data/Dataset_BUSI_with_GT/', size=tuple(params["input_dim"]), verbose=True)
        if params["verbose"]:
            print('X Train: ', np.shape(X_train))
            print('X Test: ', np.shape(X_test))
            print('y Train: ', np.shape(y_train))
            print('y Test: ', np.shape(y_test))
        
        X_train_inception, X_test_inception, y_train_inception, y_test_inception = load_data('./data/Dataset_BUSI_with_GT/', get_masks=True, size=tuple(params["input_dim"]), verbose=True)

        if params["verbose"]:
            print('X Train inception: ', np.shape(X_train_inception))
            print('X Test inception: ', np.shape(X_test_inception))
            print('y Train inception: ', np.shape(y_train_inception))
            print('y Test inception: ', np.shape(y_test_inception))

        run_Inception(X_train_inception, X_test_inception, y_train_inception, y_test_inception, file_save=params["file_save"], input_dim=tuple(params["input_dim"])+(1,), verbose=params["verbose"])
    
    # Latents dimensions comparison
    elif args.method == 'latents':
        if type(params["latent_dim"]) is list:
            list_latent = params["latent_dim"].copy()
            for latent in list_latent:
                params["latent_dim"] = latent
                X_train, X_test, y_train, y_test = load_data('./data/Dataset_BUSI_with_GT/', size=tuple(params["input_dim"]), verbose=True)
                if params["verbose"]:
                    print('X Train: ', np.shape(X_train))
                    print('X Test: ', np.shape(X_test))
                    print('y Train: ', np.shape(y_train))
                    print('y Test: ', np.shape(y_test))

                X_train_encoded, X_test_encoded = runAE(input_dim=params["input_dim"], latent_dim=params["latent_dim"], arch=params["arch"], with_cpu=params["with_cpu"], n_epochs=params["n_epochs"], data_train=X_train, data_test=X_test, file_save=params["file_save"], verbose=params["verbose"])

                run_RFC(X_train_encoded, X_test_encoded, y_train, y_test, params["file_save"], params["latent_dim"])
        else:
            raise ValueError(f"You forgot to specify the as 'latent_dim' a list of desired latent dimension to compare")
    else:
        raise ValueError(f"Not recognized {args.method} method. The availabre methods are 'both' (default), 'ae-rfc', 'inception' or 'latents'.")
        
    
if __name__ == "__main__":
    main()
    

