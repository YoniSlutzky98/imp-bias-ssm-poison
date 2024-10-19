import numpy as np

def generate_inputs(input_dim, sd_baseline, sd_special, seed, baseline_input=None, special_input=None):
    '''
    Generates input array with Gaussian entries.
    Intakes arrays which specify which entries are to be non-zero.
    '''
    np.random.seed(seed)
    if baseline_input is not None:
        n = baseline_input.shape[0]
        for i in range(n):
            nonzero_rows = np.where(np.any(baseline_input[i] != 0, axis=1))[0]
            randomized_values = np.random.normal(0, sd_baseline, size=(len(nonzero_rows), input_dim))
            baseline_input[i, nonzero_rows, :] = randomized_values
    if special_input is not None:
        m = special_input.shape[0]
        for i in range(m):
            nonzero_rows = np.where(np.any(special_input[i] != 0, axis=1))[0]
            randomized_values = np.random.normal(0, sd_special, size=(len(nonzero_rows), input_dim))
            special_input[i, nonzero_rows, :] = randomized_values
    if baseline_input is not None:
        if special_input is not None:
            return np.concatenate((baseline_input, special_input), axis=0)
        else:
            return baseline_input
    else:
        return special_input

def create_one_hot_array(length, input_dim):
    '''
    Creates a 3 axis array holding all one-hot matrices of dimensions length x input_dim.
    '''
    arr = np.zeros((length * input_dim, length, input_dim))
    indices = np.arange(length * input_dim)
    row_indices = indices // input_dim
    col_indices = indices % input_dim
    arr[indices, row_indices, col_indices] = 1
    return arr
