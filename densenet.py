#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, InputLayer, LeakyReLU, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.applications.densenet import DenseNet121
from sklearn.metrics import roc_curve, auc
from dataset_batch import load_train_data, load_test_data
import tensorflow as tf


# In[2]:


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
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)

    # Data standardization
    def standardize(self, x):
        self.core_idg.fit(x)
        return
        
    def fit(self, x, y, vx, vy):

        # fit the data
        print ("Start Training model")
        X_train, y_train, X_test, y_test = x, y, vx, vy
        hist = self.model.fit_generator(
            (self.core_idg.flow(X_train, y_train, batch_size = 32)),
            validation_data = self.core_idg.flow(X_test, y_test), epochs=self.epochs)
        
        print ("Done Training model")
        print ("AVG AUC:", self.score(X_test, y_test))
        return hist
    
    # data did preprocessing            
    def inference(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        
        # make predicition
        y_pred = self.predict(x)
        auc_val = 0.0

        # calculate score
        for idx in range(self.output_dim):
            fpr, tpr, _ = roc_curve(y[:,idx].astype(int), y_pred[:,idx])
            auc_val += auc(fpr, tpr)
        
        return auc_val / self.output_dim

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


# In[3]:


if __name__ == "__main__":
    
    model = DenseNetModel(input_dim=(224,224,3), epochs=5)

    X_vali, y_vali = load_train_data(min_cnt=9500, max_cnt=10002)
    for iter in range(10):
        for base in range(0, 9500, 500):
            X_train, y_train = load_train_data(min_cnt=base, max_cnt=base+499)
            model.standardize(X_train)
            model.fit(X_train, y_train, X_vali, y_vali)

    # Saving model
    model.save_weight()


# In[ ]:




