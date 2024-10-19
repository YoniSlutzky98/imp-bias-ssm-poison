from tensorflow import keras
from .constraint import DiagonalConstraint
from .scheduler import AdaptiveLearningRateScheduler

def create_ssm(state_dim, length, seed, sd_A, sd_B_C, base_lr, beta=0.8, soft_const=1e-6, adaptive=False, mlp_dim=0,
               depth=0, sd_D=0):
    '''
    Creates the SSM model (either standalone or followed by a non-linear MLP).
    '''
    default_kernel_initializer_ssm = keras.initializers.RandomNormal(mean=0., stddev=sd_B_C, seed=seed)
    default_recurrent_initializer = keras.initializers.RandomNormal(mean=0., stddev=sd_A, seed=seed)
    default_kernel_initializer_dense = keras.initializers.RandomNormal(mean=0., stddev=sd_B_C, seed=seed)
    model = keras.Sequential()
    constraint = DiagonalConstraint()
    ssm_layer = keras.layers.SimpleRNN(state_dim, input_shape=(length, 1), use_bias=False,
                                       return_sequences=False, activation='linear',
                                       kernel_initializer=default_kernel_initializer_ssm,
                                       recurrent_initializer=default_recurrent_initializer,
                                       recurrent_constraint=constraint)
    model.add(ssm_layer)
    dense_layer = keras.layers.Dense(1, use_bias=False, kernel_initializer=default_kernel_initializer_dense,
                                     activation='linear')
    model.add(dense_layer)
    if mlp_dim > 0:
        default_kernel_initializer_mlp = keras.initializers.RandomNormal(mean=0., stddev=sd_D, seed=seed)
        for _ in range(depth):
            mlp_layer = keras.layers.Dense(mlp_dim, use_bias=False, kernel_initializer=default_kernel_initializer_mlp,
                                           activation='relu')
            model.add(mlp_layer)
        out_layer = keras.layers.Dense(1, use_bias=False, kernel_initializer=default_kernel_initializer_mlp,
                                       activation='linear')
        model.add(out_layer)

    loss_fn = keras.losses.MeanSquaredError()
    if adaptive:
        scheduler = AdaptiveLearningRateScheduler(base_lr, beta, soft_const, model)
        optimizer = keras.optimizers.SGD(learning_rate=scheduler)
    else:
        scheduler = None
        optimizer = keras.optimizers.Adam(learning_rate=base_lr)

    model.compile(optimizer=optimizer, loss=loss_fn)
    return model, scheduler

def get_ssm_weights(model):
    '''
    Returns the weights of the SSM model.
    '''
    W = model.get_weights()
    B, A, C = W[0], W[1], W[2]
    if len(W) > 3:
        D = W[3:len(W)]
        return A, B, C, D
    else:
        return A, B, C

def set_ssm_weights(model, W):
    '''
    Sets the weights of the SSM model.
    '''
    model.layers[0].set_weights((W[1], W[0]))
    model.layers[1].set_weights([W[2]])
    if len(W) > 3:
        for i in range(len(W[3])):
            model.layers[2+i].set_weights([W[3][i]])