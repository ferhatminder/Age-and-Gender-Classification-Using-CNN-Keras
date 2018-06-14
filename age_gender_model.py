import os
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import utils as myutils


class AgeGenderModel(object):
    def __init__(self, model_variables, variables, database):
        self.MV = model_variables
        self.Vars = variables
        self.DB = database
        self.model = None
        self.weight_save_path = "saved_weights"
        self.outputs_path = "outputs"
        self.model_name = self.Vars["name"]

        if self.Vars["class"] == "age":
            self.model_class = "age"
            self.class_count = self.DB.age_class_count
            self.class_labels = self.DB.age_labels
            self.db_train_path = self.DB.db_age_train_folder_path
            self.db_test_path = self.DB.db_age_test_folder_path
            self.mean_image = myutils.load_image( self.DB.age_mean_image_path )
        elif self.Vars["class"] == "sex":
            self.model_class = "sex"
            self.class_count = self.DB.sex_class_count
            self.class_labels = self.DB.sex_labels
            self.db_train_path = self.DB.db_sex_train_folder_path
            self.db_test_path = self.DB.db_sex_test_folder_path
            self.mean_image = myutils.load_image( self.DB.sex_mean_image_path )

        self.class_weights = myutils.load_obj(
            self.DB.db_new_path + "/" + self.model_class)

    def create_model(self):
        self.model = Sequential([
            Conv2D(
                self.MV["conv1"]["filter_size"],
                tuple(self.MV["conv1"]["kernel_size"]),
                strides=tuple(self.MV["conv1"]["stride"]),
                padding=self.MV["conv1"]["padding"],
                bias_initializer=self.MV["conv1"]["bias"],
                activation='relu',
                input_shape=self.DB.input_shape),
            MaxPooling2D(
                pool_size=tuple(self.MV["pool1"]["pool_shape"]),
                strides=tuple(self.MV["pool1"]["stride"]),
                padding=self.MV["pool1"]["padding"]),

            #    Used batch normalization instead of local response normalization
            BatchNormalization(),
            Conv2D(
                self.MV["conv2"]["filter_size"],
                tuple(self.MV["conv2"]["kernel_size"]),
                strides=tuple(self.MV["conv2"]["stride"]),
                padding=self.MV["conv2"]["padding"],
                bias_initializer=self.MV["conv2"]["bias"],
                activation='relu'),
            MaxPooling2D(
                pool_size=tuple(self.MV["pool2"]["pool_shape"]),
                strides=tuple(self.MV["pool2"]["stride"]),
                padding=self.MV["pool2"]["padding"]),

            #    Used batch normalization instead of local response normalization
            BatchNormalization(),
            Conv2D(
                self.MV["conv3"]["filter_size"],
                tuple(self.MV["conv3"]["kernel_size"]),
                strides=tuple(self.MV["conv3"]["stride"]),
                padding=self.MV["conv3"]["padding"],
                bias_initializer=self.MV["conv3"]["bias"],
                activation='relu'),
            MaxPooling2D(
                pool_size=tuple(self.MV["pool3"]["pool_shape"]),
                strides=tuple(self.MV["pool3"]["stride"]),
                padding=self.MV["pool3"]["padding"]),
            Flatten(),
            Dense(
                self.MV["dense1"], bias_initializer="ones", activation='relu'),
            Dropout(self.MV["drop1"]),
            Dense(
                self.MV["dense2"], bias_initializer="ones", activation='relu'),
            Dropout(self.MV["drop2"]),
            Dense(
                self.class_count,
                bias_initializer="ones",
                activation='softmax')
        ])

        adam = Adam(lr=self.MV["learning_rate"], decay=self.MV["decay"])

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])

    def init_image_data_generators(self):
        self.train_idgen = ImageDataGenerator(
            featurewise_center=True,
            rescale=1. / 255,
            rotation_range=10,
            height_shift_range=0.10,
            width_shift_range=0.10,
            horizontal_flip=True)

        self.test_idgen = ImageDataGenerator(
            rescale=1. / 255,
            featurewise_center=True)
        
        mean_img_norm = np.asarray(self.mean_image,np.float32)
        mean_img_norm /= 255 
        self.train_idgen.mean = mean_img_norm
        self.test_idgen.mean = mean_img_norm

        self.train_idgen_flow = self.train_idgen.flow_from_directory(
            self.db_train_path,
            classes=self.class_labels,
            target_size=(int(self.Vars["img_in_height"]),
                         int(self.Vars["img_in_width"])),
            batch_size=int(self.Vars["batch_size"]),
            class_mode='categorical',
            shuffle=True)

        self.test_idgen_flow = self.test_idgen.flow_from_directory(
            self.db_test_path,
            classes=self.class_labels,
            target_size=(int(self.Vars["img_in_height"]),
                         int(self.Vars["img_in_width"])),
            batch_size=int(self.Vars["batch_size"]),
            class_mode='categorical')

    def train(self):
        self.model.fit_generator(
            self.train_idgen_flow,
            steps_per_epoch=self.train_idgen_flow.n //
            self.train_idgen_flow.batch_size,
            epochs=int(self.Vars["epoch"]),
            class_weight=self.class_weights,
#            validation_data=self.test_idgen_flow,
#            validation_steps= self.test_idgen_flow.n//self.test_idgen_flow.batch_size ,
            workers=1,
            verbose=2)

    def test(self):
        self.test_idgen_flow.reset()
        score = self.model.evaluate_generator(
            self.test_idgen_flow,
            self.test_idgen_flow.n // self.test_idgen_flow.batch_size)
        print("Test Score")        
        print("Loss: {:.3f}\tAccuracy: {:.3f}".format(score[0], score[1]))

    def predict_test(self):
        self.test_idgen_flow.reset()
        predictions = np.empty(shape=[0, self.class_count])
        true_classes = np.empty(shape=[0, self.class_count])
        for i in range(self.test_idgen_flow.n//self.test_idgen_flow.batch_size):
            X_batch , Y_batch = self.test_idgen_flow.next()
            preds= self.model.predict(X_batch,batch_size=self.test_idgen_flow.batch_size)
            
            true_classes = np.append(true_classes, Y_batch, axis=0)
            predictions = np.append(predictions, preds, axis=0)
     
        return predictions,true_classes

    def save_model_weights(self):
        if not os.path.exists(self.weight_save_path):
            os.makedirs(self.weight_save_path)
        path = self.weight_save_path + "/" + self.Vars["name"] + ".h5"
        self.model.save_weights(path)

    def load_model_weights(self):
        path = self.weight_save_path + "/" + self.Vars["name"] + ".h5"
        self.model.load_weights(path)


def print_preds(agm, predictions, true_values, accuracy, my_accuracy, f1sc):
    from utils import log_preds
    log_preds(
        "Path\t\t\t\t True Class\t Predicted Class\tACCURACY = {:.3f}\tMY ACCURACY = {:.3f}\tF1 Score = {:.3f}".
        format(accuracy, my_accuracy, f1sc), agm.model_name)
    for i in range(len(predictions)):

        log_str = "{}\t {}\t".format(agm.test_idgen_flow.filenames[i],
                                     true_values[i])
        for j in range(len(predictions[0]) - 1):
            log_str += "{:.2f}\t".format(predictions[i][j])
        log_str += "{:.2f}".format(predictions[i][len(predictions[0]) - 1])
        log_preds(log_str, agm.model_name)


# Just for debug
#from keras.preprocessing.image import ImageDataGenerator
#import numpy as np
#
#train_idgen = ImageDataGenerator(
#        featurewise_std_normalization=True
#        ,rescale=1. / 255
#        ,shear_range=0.10
#        ,rotation_range=20
#        ,height_shift_range=0.10
#        ,width_shift_range=0.10
#        ,horizontal_flip=True
#        )
#
#sample_data_gen_flow = ImageDataGenerator(rescale= 1. /255).flow_from_directory(
#        "DB/age/train",
#        target_size=(227,227),
#        batch_size= 64,
#        class_mode='categorical',
#        shuffle=True)
#
#X , _ = sample_data_gen_flow.next()
#for i in range( 1000 // 64 ):
#    X2 , _ = sample_data_gen_flow.next()
#    X = np.append(X,X2, axis=0)
#
#train_idgen.fit(X)
#
#train_idgen_flow = train_idgen.flow_from_directory(
#        "DB/age/train",
#        target_size=(227,227),
#        batch_size=64,
#        class_mode='categorical'
#        ,save_to_dir="outputs\\aug_train"
#        )
#train_idgen_flow.next()
#train_idgen_flow.reset()
