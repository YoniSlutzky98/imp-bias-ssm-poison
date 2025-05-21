import numpy as np
from tensorflow import keras, GradientTape
from .model import create_ssm, get_ssm_weights, set_ssm_weights

class StoppingCallback(keras.callbacks.Callback):
    def __init__(self, model, inputs, outputs, eps):
        super(StoppingCallback, self).__init__()
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.eps = eps
        self.opt_index = -1

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model(self.inputs)
        loss = np.mean((preds - self.outputs) ** 2)
        if loss is not None and loss <= self.eps:
            self.model.stop_training = True
            self.opt_index = epoch
            print("+------------------------------+")
            print("|Reached sub-epsilon train loss|")
            print("+------------------------------+")

class LoggingCallback(keras.callbacks.Callback):
    '''
    Callback that logs losses and diagonal entries during training.
    Logs status every log_period epochs, and prints status every print_period epochs.
    Potentially enforces fixing of B/C parameter matrices to being all ones (higher rank standalone experiments).
    '''
    def __init__(self, model, train_inputs, train_outputs, ext_inputs, ext_outputs, log_period=100, print_period=10000,
                 n_evals=7, mlp_dim=0, depth=0, exper_type='dynamics', fix_B_C=False):
        super(LoggingCallback, self).__init__()
        self.model = model
        self.length = train_inputs.shape[1]
        self.state_dim = get_ssm_weights(self.model)[0].shape[0]
        self.ext_model, _ = create_ssm(self.state_dim, ext_inputs.shape[1], 0, 1, 1, 0,
                                       mlp_dim=mlp_dim, depth=depth)
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.ext_inputs = ext_inputs
        self.ext_outputs = ext_outputs
        self.log_period = log_period
        self.print_period = print_period
        self.n_evals = min(n_evals, self.state_dim)
        self.exper_type = exper_type
        self.fix_B_C = fix_B_C
        self.train_losses = []
        self.ext_losses = []
        self.evals = []
        self.gammas = []
        self.stamps = []

    def on_epoch_end(self, epoch, logs=None):
        W = list(get_ssm_weights(self.model))
        if self.fix_B_C:
            W[1] = np.ones(W[1].shape)
            W[2] = np.ones(W[2].shape)
            set_ssm_weights(self.model, W)

        if epoch % self.log_period == 0:
            self.stamps.append(epoch)
            train_preds = self.model(self.train_inputs)
            train_loss = np.mean((train_preds - self.train_outputs) ** 2)
            self.train_losses.append(train_loss)

            set_ssm_weights(self.ext_model, W)
            ext_preds = self.ext_model(self.ext_inputs)
            if len(W) > 3: # SSM in non-linear neural network
                ext_loss = np.linalg.norm(ext_preds - self.ext_outputs) / np.linalg.norm(self.ext_outputs)
            else: # standalone SSM
                ext_loss = np.max(np.abs(ext_preds - self.ext_outputs)) / np.max(np.abs(self.ext_outputs))
            self.ext_losses.append(ext_loss)

            evals = np.linalg.eigvals(W[0])
            evals = np.abs(evals[np.argsort(np.abs(evals))[::-1]])
            self.evals.append(evals)

            deltas = self.train_outputs - train_preds
            gamma = np.zeros(self.length - 1)
            if len(W) > 3: # SSM in non-linear neural network
                f_model = keras.Model(inputs=self.model.input, outputs=self.model.layers[1].output)
                g_input = keras.Input(shape=f_model.output_shape[1:])
                g_output = g_input
                for layer in self.model.layers[2:]:
                    g_output = layer(g_output)
                g_model = keras.Model(inputs=g_input, outputs=g_output)
                z = f_model(self.train_inputs)
                with GradientTape() as tape:
                    tape.watch(z)
                    g_z = g_model(z)
                ksis = tape.gradient(g_z, z).numpy()
                for l in range(self.length - 1):
                    gamma[l] = (2 * (l + 1) / self.train_inputs.shape[0]) * np.sum(
                        deltas * ksis * self.train_inputs[:, self.length - l - 2])
            else: # standalone SSM
                for l in range(self.length - 1):
                    gamma[l] = (2 * (l + 1) / self.train_inputs.shape[0]) * np.sum(
                        deltas * self.train_inputs[:, self.length - l - 2])
            self.gammas.append(gamma)

        if epoch % self.print_period == 0:
            print("-------------------------------------------------------")
            print("-------------------------------------------------------")
            print(f'Epoch: {epoch}')
            print(f'Train loss: {self.train_losses[-1]}')
            if self.exper_type == 'dynamics':
                print(f'{self.n_evals} absolute largest EVs of A: {self.evals[-1][:self.n_evals]}')

class GradientNormCallback(keras.callbacks.Callback):
    '''
    Auxiliary callback that computes gradient norms.
    '''
    def __init__(self, model, inputs, outputs, period):
        super(GradientNormCallback, self).__init__()
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period == 0:
            # current_weights = self.model.get_weights()
            with GradientTape() as tape:
                predictions = self.model(self.inputs)
                loss = self.model.loss(predictions, self.outputs)
            gradients = tape.gradient(loss, self.model.trainable_weights)
            gradient_norms = [np.linalg.norm(grad.numpy()) for grad in gradients]
            print()
            for i, norm in enumerate(gradient_norms):
                print(f'Epoch {epoch + 1}, Weight {i} Gradient Norm: {norm:.15f}')