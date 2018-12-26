#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy.random import seed
seed(3363)
from tensorflow import set_random_seed
set_random_seed(3363)


# In[ ]:


from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, InputLayer, LeakyReLU, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.applications.densenet import DenseNet121
from sklearn.metrics import roc_auc_score
from dataset_batch import load_train_data, load_test_data
import tensorflow as tf


# In[ ]:


def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

class DenseNetModel():

    def __init__(self, input_dim=(224,224,3), output_dim=14, learning_rate=0.00001, epochs=5, drop_out=0.3):

        # parms:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.drop_out = drop_out

        # Define DenseNet
        self.model = Sequential()

        base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=self.input_dim)

        # Freeze the model first
        base_model.trainable = False

        self.model.add(base_model)

        self.model.add(Dropout(self.drop_out))
        self.model.add(Dense(512))
        self.model.add(Dropout(self.drop_out))
        self.model.add(Dense(self.output_dim, activation = 'sigmoid'))

        auc_roc = as_keras_metric(tf.metrics.auc)
        recall = as_keras_metric(tf.metrics.recall)
        precision = as_keras_metric(tf.metrics.precision)

        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae', auc_roc, recall, precision])
        self.model.summary()

        # Image augmentation
        self.core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip=True, 
                              vertical_flip=False, 
                              height_shift_range=0.05, 
                              width_shift_range=0.15, 
                              rotation_range=10, 
                              shear_range=0.2,
                              fill_mode='constant',
                              cval=1,
                              zoom_range=[0.85,1.20])

    # Data standardization
    def standardize(self, x):
        self.core_idg.fit(x)
        return

    # fit the data
    def fit(self, x, y, vx, vy):
        print ("Start Training model")
        X_train, y_train, X_test, y_test = x, y, vx, vy
        hist = self.model.fit_generator(
            (self.core_idg.flow(X_train, y_train, batch_size = 25)),
            validation_data = self.core_idg.flow(X_test, y_test), epochs=self.epochs)
        print ("Done Training model")
        return hist

    # data did preprocessing            
    def inference(self, x):
        return self.model.predict(x)

    # make predicition
    def score(self, x, y):
        y_pred = self.predict(x)
        return roc_auc_score(y, y_pred, average = "macro")

    # pain data without preprocessing
    def predict(self, x):
        x = self.core_idg.standardize(x)
        return self.inference(x)

    def save_weight(self, path = 'baseline.h5'):
        print ("Start Saving model")
        self.model.save(path)
        print ("Done Saving model")
        return

    def load_weight(self, path = 'baseline.h5'):
        print ("Start Loading model")
        self.model.load_weights(path)
        print ("Done Loading model")
        return


# In[ ]:


if __name__ == "__main__":

    model = DenseNetModel(input_dim=(224,224,3), epochs=5, drop_out=0.4)

    X_vali, y_vali = load_train_data(min_cnt=0, max_cnt=800)
    for itera in range(40):
        for base in range(800, 10001, 2400):
            X_train, y_train = load_train_data(min_cnt=base, max_cnt=base+2399)
            model.standardize(X_train)
            model.fit(X_train, y_train, X_vali, y_vali)
        print("Iter",itera,"AVG AUC:",model.score(X_vali, y_vali))

    # Saving model
    model.save_weight()


# In[ ]:




