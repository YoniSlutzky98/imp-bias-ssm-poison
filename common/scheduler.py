import tensorflow as tf
from tensorflow import keras

class AdaptiveLearningRateScheduler(keras.optimizers.schedules.LearningRateSchedule):
    '''
    Adaptive learning rate scheduler from appendix D.2 of https://arxiv.org/abs/2201.11729.
    '''
    def __init__(self, base_lr, beta, soft_const, model):
        super(AdaptiveLearningRateScheduler, self).__init__()
        self.base_lr = base_lr
        self.beta = beta
        self.soft_const = soft_const
        self.gamma = tf.Variable(0.0, trainable=False)
        self.model = model
        self.inputs = None
        self.outputs = None

    def set_examples(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, step):
        gamma = self.beta * self.gamma + (1 - self.beta) * self.compute_gradient_norm()
        self.gamma.assign(gamma)
        return self.base_lr / ((gamma / (1 - self.beta ** (step + 1))) ** 0.5 + self.soft_const)

    def compute_gradient_norm(self):
        with tf.GradientTape() as tape:
            predictions = self.model(self.inputs)
            loss = self.model.loss(predictions, self.outputs)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        gradient_norm_sum = tf.math.reduce_sum([tf.norm(grad) for grad in gradients])
        del predictions, loss, gradients, tape
        return gradient_norm_sum

    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'beta': self.beta
        }
