# -*- coding: utf-8 -*-

#from tensorflow.keras import backend as K
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 