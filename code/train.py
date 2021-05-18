# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, MaxPooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16, VGG19, InceptionV3, NASNetMobile, ResNet50V2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os


IMG_WIDTH = 32
IMG_HEIGHT = 32
BATCH_SIZE = 64
CLASSES = 43
EXP_NAME = 'TRAFFIC-SIGN-CLASSIFIER'
CLASS_MODE = "categorical"


def _append_data_to_file_(file_path, data):
    try:
        with open(file_path, 'a+') as f:
            f.write(data + "\n")
    except Exception as e:
        print(e)


def plot_training(H, N, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


def train(network_name='mobileNetV2', experiment_number=1, out_dir="..//data//"):

    EXPERIMENT_NUMBER = experiment_number
    EXPERIMENT_INFO = 'training '+ network_name + ' on German Traffic Sign Dataset'

    # derive the paths to the training, validation, and testing
    train_data_dir = '//data//CNN//train'
    valid_data_dir = '//data//CNN//val'
    test_data_dir = '//data//CNN//test'

    featurewise_center = False  # set input mean to 0 over the dataset
    samplewise_center = False # set each sample mean to 0
    featurewise_std_normalization = False  # divide inputs by std of the dataset
    samplewise_std_normalization = False  # divide each input by its std
    zca_whitening = False  # apply ZCA whitening
    zca_epsilon = 1e-06  # epsilon for ZCA whitening
    rotation_range = 5  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range = 0.1# randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.1 # randomly shift images vertically (fraction of total height)
    shear_range = 0.2  # set range for random shear
    zoom_range = 0.2  # set range for random zoom
    channel_shift_range = 0.  # set range for random channel shifts
    fill_mode = 'nearest  '# set mode for filling points outside the input boundaries
    cval = 0.  # value used for fill_mode = "constant"
    horizontal_flip = True # randomly flip images
    vertical_flip = False  # randomly flip images
    rescale = 1.0/255 # set rescaling factor (applied before any other transformation)
    preprocessing_function = None # set function that will be applied on each input

    trainAug = ImageDataGenerator( featurewise_center=featurewise_center,
                                   samplewise_center=samplewise_center,
                                   featurewise_std_normalization=featurewise_std_normalization,
                                   samplewise_std_normalization=samplewise_std_normalization,
                                   zca_whitening=zca_whitening ,zca_epsilon=zca_epsilon,
                                   rotation_range=rotation_range,
                                   width_shift_range=width_shift_range ,height_shift_range=height_shift_range,
                                   shear_range=shear_range ,zoom_range=zoom_range,
                                   channel_shift_range=channel_shift_range,
                                   fill_mode=fill_mode ,cval=cval, horizontal_flip=horizontal_flip,
                                   vertical_flip=vertical_flip ,rescale=rescale,
                                   preprocessing_function=preprocessing_function )


    valAug = ImageDataGenerator(rescale=rescale)

    # load and iterate training dataset
    train_generator = trainAug.flow_from_directory(train_data_dir, class_mode=CLASS_MODE
                                                   ,target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                   batch_size=BATCH_SIZE ,color_mode = 'rgb' ,shuffle=True
                                                   ,seed=42)
    # load and iterate validation dataset,
    val_generator = valAug.flow_from_directory(valid_data_dir, class_mode=CLASS_MODE
                                               ,target_size=(IMG_WIDTH, IMG_HEIGHT),
                                               batch_size=BATCH_SIZE ,color_mode = 'rgb' ,shuffle=False ,seed=42)
    # load and iterate test dataset
    test_generator = valAug.flow_from_directory(test_data_dir ,class_mode=CLASS_MODE
                                                ,target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                batch_size=BATCH_SIZE ,color_mode = 'rgb' ,shuffle=False ,seed=42)


    print('img size: ', IMG_WIDTH ,'x', IMG_HEIGHT)
    input_tensor = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

    if network_name == "mobileNetV2":
        baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor, input_shape=input_shape)
    elif network_name == "vgg16":
        baseModel = VGG16(weights="imagenet", include_top =False, input_tensor =input_tensor, input_shape=input_shape)
    elif network_name == "vgg19":
        baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=input_tensor, input_shape=input_shape)
    elif network_name == "InceptionV3":
        baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=input_tensor, input_shape=input_shape)
    elif network_name == "NasNetMobile":
        baseModel = NASNetMobile(input_shape=input_shape ,include_top=False, weights=None ,input_tensor=input_tensor,classes=2)
    elif network_name == "resNet50V2":
        baseModel = ResNet50V2(weights="imagenet", include_top=False, input_tensor=input_tensor, input_shape=input_shape)
    else:
        print("No model given therefore selecting VGG16\n")
        baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=input_tensor, input_shape=input_shape)

    for layer in baseModel.layers:
        layer.trainable= False

    print(baseModel.summary())

    """
    Upon experimenting,in case of VGG16, it was observed that using the model till block4_pool layer was sufficient with the available dataset so as to reduce over-fitting
    """

    last_layer = baseModel.get_layer('block4_pool')
    headModel = last_layer.output

    # headModel = Conv2D(60,(5,5))(headModel)
    # headModel = Conv2D(60,(3,3))(headModel)

    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(240, activation="relu")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(120, activation="relu")(headModel)
    headModel = Dense(CLASSES, activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    print(model.summary())

    model_name = network_name +'_sgd'

    details = {"model_name": model_name, "optimizer": {"name": 'sgd ' +" default"},
               "dataset": EXP_NAME, "info": "we will train for 500 epochs or earlystopping " +EXPERIMENT_INFO}


    print("=> Going to train model: ", model_name)
    print(model.metrics_names)

    # initialize paths
    data_dir = os.path.join(out_dir+"//RESULTS", EXP_NAME)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    experiment_dir = os.path.join(data_dir, str(EXPERIMENT_NUMBER), model_name)
    weights_dir = os.path.join(experiment_dir, "weights")

    # create directories
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    history_file = os.path.join(experiment_dir, "history")
    plot_file_dir = experiment_dir

    details_file = os.path.join(experiment_dir, "details")
    logs_file = os.path.join(experiment_dir, "logs")

    # create directories
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)


    # initialize callbacks
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=25, mode='min', verbose=1)

    best_loss_model_checkpoint_path = os.path.join(weights_dir, "best_loss_weights.hdf5")
    best_loss_checkpoint = ModelCheckpoint(filepath=best_loss_model_checkpoint_path, monitor='loss',
                                           save_best_only=True, verbose=1, mode='min')

    best_val_loss_model_checkpoint_path = os.path.join(weights_dir, "best_val_loss_weights.hdf5")
    best_val_loss_checkpoint = ModelCheckpoint(filepath=best_val_loss_model_checkpoint_path, monitor='val_loss',
                                               save_best_only=True, verbose=1, mode='min')

    callbacks = [best_loss_checkpoint, best_val_loss_checkpoint, early_stop ]

    # train model
    time_stamp = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    _append_data_to_file_(logs_file, "Training Started at "+time_stamp)

    _append_data_to_file_(logs_file, "Starting training on  data")

    print("[INFO] compiling model...")
    opt = SGD(lr=1e-4, momentum=0.9)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=int(train_generator.n / BATCH_SIZE),
                        epochs=500,
                        callbacks=callbacks,
                        validation_data=val_generator,
                        validation_steps=int(val_generator.n / BATCH_SIZE))

    _append_data_to_file_(logs_file, "Done training on data")

    time_stamp = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    _append_data_to_file_(logs_file, "Training Ended at "+time_stamp)

    print("=> Model training done for: ", model_name)

    print("=> Evaluating model: ", model_name)
    score = model.evaluate_generator(test_generator, steps=24)

    print('loss:', score[0])
    print('accuracy:', score[1])


def save_model(base_folder):
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
    input_tensor = Input(shape=input_shape)
    weights_path = base_folder + "/weights/best_val_loss_weights.hdf5"

    baseModel = MobileNetV2(weights=None, include_top=False, input_tensor=input_tensor, input_shape=input_shape)

    last_layer = baseModel.get_layer('block4_pool')
    headModel = last_layer.output

    # headModel = Conv2D(60,(5,5))(headModel)
    # headModel = Conv2D(60,(3,3))(headModel)

    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(240, activation="relu")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(120, activation="relu")(headModel)

    headModel = Dense(CLASSES, activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
    model.load_weights(weights_path)
    model.save(base_folder + "/weights/best_val_loss_weights.h5")
    print("model saved")


def freeze_graph(model_file_path, save_pb_dir, save_pb_name="froze_model.pb", save_pb_as_text=False):
    from tensorflow.keras.models import load_model
    from tensorflow.python.framework import graph_io

    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)

    model = load_model(model_file_path)
    session = tf.keras.backend.get_session()

    INPUT_NODE = [t.op.name for t in model.inputs]
    OUTPUT_NODE = [t.op.name for t in model.outputs]
    print(INPUT_NODE, OUTPUT_NODE)

    graph = session.graph
    output = [out.op.name for out in model.outputs]

    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        print("graph saved")


if __name__ == '__main__':

    out_dir = "..//data//"
    """models to be used for training, change experiment number accordingly"""
    # train(1, "mobileNetV2",out_dir)
    train(2, "vgg16", out_dir)


    """
    After training is complete
    """
    # base_folder = out_dir+"RESULTS/"+EXP_NAME+"/2/vgg16_sgd"
    # weights_folder = base_folder+"/weights"
    # h5_val_weights = weights_folder + "/best_val_loss_weights.h5"
    # pb_filename = base_folder.split('/')[-2]+"_"+base_folder.split('/')[-1]+"_trafficSign.pb"
    # print(pb_filename)
    # save_model(base_folder)
    # freeze_graph(h5_val_weights, weights_folder, pb_filename)
