import tensorflow as tf
from tensorflow import keras

class DiagonalConstraint(keras.constraints.Constraint):
    def __init__(self):
        super(DiagonalConstraint, self).__init__()

    def __call__(self, w):
        diag = tf.linalg.diag(tf.linalg.diag_part(w))
        return diag

    def get_config(self):
        return {}
