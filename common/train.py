import numpy as np
from .model import create_ssm, get_ssm_weights, set_ssm_weights
from .callbacks import StoppingCallback, LoggingCallback, GradientNormCallback
from .plotting import plot_func

def train_helper(train_inputs, train_outputs, ext_inputs, ext_outputs, state_dim, seed, sd_A, sd_B_C, base_lr, epochs,
                 eps, diff=0, warm_init=0, beta=0.8, soft_const=1e-6, adaptive=False, mlp_dim=0, depth=0, sd_D=0,
                 log_period=100, print_period=10000, n_evals=7, epochs_after_opt=0, track_gradients=False,
                 exper_type='dynamics', fix_B_C=False, dim3=False, dim4=False):
    np.random.seed(seed)
    length = train_inputs.shape[1]
    batch_size = train_outputs.shape[0]

    model, scheduler = create_ssm(state_dim, length, seed, sd_A, sd_B_C, base_lr, beta=beta, soft_const=soft_const,
                                  adaptive=adaptive, mlp_dim=mlp_dim, depth=depth, sd_D=sd_D)

    W = list(get_ssm_weights(model))
    A, B, C = W[0], W[1], W[2]
    A = np.diag(np.flip(np.sort(np.abs(np.diag(A)))))
    B = np.flip(np.sort(np.abs(B)))
    C = np.flip(np.sort(np.abs(C)))
    A[1, 1] = A[0, 0] - diff
    B[0, 1] = B[0, 0] - diff
    if dim3 or dim4:
        A[2, 2] = A[0, 0] - 1.01 * diff
        B[0, 2] = B[0, 0] - 1.01 * diff
    if dim4:
        A[3, 3] = A[0, 0] - 1.05 * diff
        B[0, 3] = B[0, 0] - 1.05 * diff
    A = A + warm_init * np.eye(state_dim)
    B = B + warm_init * np.ones(B.shape)
    W[0], W[1], W[2] = A, B, C
    set_ssm_weights(model, W)

    if adaptive:
        scheduler.set_examples(train_inputs, train_outputs)
    stopping_cb = StoppingCallback(model, train_inputs, train_outputs, eps)
    logging_cb = LoggingCallback(model, train_inputs, train_outputs, ext_inputs, ext_outputs, log_period=log_period,
                                 print_period=print_period, n_evals=n_evals, mlp_dim=mlp_dim, depth=depth,
                                 exper_type=exper_type, fix_B_C=fix_B_C)
    cb = [stopping_cb, logging_cb]
    if track_gradients:
        gradient_cb = GradientNormCallback(model, train_inputs, train_outputs, period=print_period)
        cb.append(gradient_cb)

    model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=cb)

    post_opt_logging_cb = LoggingCallback(model, train_inputs, train_outputs, ext_inputs, ext_outputs,
                                          log_period=log_period, print_period=print_period, n_evals=n_evals,
                                          mlp_dim=mlp_dim, depth=depth, exper_type=exper_type, fix_B_C=fix_B_C)
    post_opt_cb = [post_opt_logging_cb]
    if track_gradients:
        gradient_cb = GradientNormCallback(model, train_inputs, train_outputs, period=print_period)
        post_opt_cb.append(gradient_cb)

    model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=epochs_after_opt, verbose=0,
              callbacks=post_opt_cb)

    print("+-------------+")
    print("|Final results|")
    print("+-------------+")
    print(f'Train loss: {post_opt_logging_cb.train_losses[-1]}')
    if exper_type == 'dynamics':
        print(f'{n_evals} absolute largest EVs of A: {post_opt_logging_cb.evals[-1]}')
    elif exper_type == 'poison':
        print(f'Ext. loss: {post_opt_logging_cb.ext_losses[-1]}')
    print("------------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------------")

    return (np.array(logging_cb.train_losses + post_opt_logging_cb.train_losses),
            np.array(logging_cb.ext_losses + post_opt_logging_cb.ext_losses),
            np.abs(np.array(logging_cb.evals + post_opt_logging_cb.evals)),
            np.array(logging_cb.gammas + post_opt_logging_cb.gammas),
            np.array(logging_cb.stamps + [stamp + logging_cb.stamps[-1] for stamp in post_opt_logging_cb.stamps]),
            stopping_cb.opt_index)

def train(train_inputs, train_outputs, ext_inputs, ext_outputs, state_dim, seed, sd_A, sd_B_C, base_lr, epochs, eps,
          diff=0, warm_init=0, beta=0.8, soft_const=1e-6, adaptive=False, mlp_dim=0, depth=0, sd_D=0, log_period=100,
          print_period=10000, n_evals=7, epochs_after_opt=0, track_gradients=False, exper_type='dynamics',
          fix_B_C=False, dim3=False, dim4=False, title=""):
    train_losses, ext_losses, evals, gammas, stamps, opt_index = train_helper(train_inputs, train_outputs, ext_inputs,
                                                                              ext_outputs, state_dim, seed, sd_A, sd_B_C,
                                                                              base_lr, epochs, eps, diff, warm_init, beta,
                                                                              soft_const, adaptive, mlp_dim, depth, sd_D,
                                                                              log_period, print_period, n_evals,
                                                                              epochs_after_opt, track_gradients, exper_type,
                                                                              fix_B_C, dim3, dim4)
    if exper_type == 'dynamics':
        plot_func(evals, gammas, stamps, n_evals, title)
    else:
        return train_losses[-1], ext_losses[-1]
