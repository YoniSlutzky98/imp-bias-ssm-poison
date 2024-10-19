# The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels
Official implementation for the experiments
in [The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](https://www.arxiv.org/abs/2410.10473), 
based on the [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/) and [SciPy](https://scipy.org/) libraries.

## Installing Requirements

Tested with Python 3.9. The ```requirements.txt``` file includes the required dependencies, which can be installed via:

```
pip install -r requirements.txt
```

## Experiments

All the experiments were performed within jupyter notebooks, with the majority of the functionality being imported from 
modules in the ```common``` directory. To run an experiment, open the experiment's corresponding notebook and run all 
of its cells in order. The experiments were carried out using a single Nvidia RTX 2080 Ti GPU. All notebooks (except 
```standalone_ssm_theory_poison_experiments.ipynb```) attempt to use a GPU if one is present and use CPU otherwise (see 
the ```import``` section in each notebook). 

### Running Dynamical Characterization Experiments (Section 4.1)

The experiments train SSM models to fit two datasets (either with or without special sequences), and then plot the 
evolution of the diagonal entries of the learned state transition matrices A. Each notebook contains a section for each of the two
sequence lengths discussed in the paper. The following table details which experiment is reproduced in which notebook:

| Experiment                                         | Notebook                                                    |
|----------------------------------------------------|-------------------------------------------------------------|
| Standalone SSM (Figures 2 and 3)                   | ```standalone_ssm_dynamics_experiments.ipynb```             |
| SSM in non-linear neural network (Figures 2 and 3) | ```ssm_in_nn_dynamics_experiments.ipynb```                  |
| Standalone SSM higher rank (Figure 4)              | ```standalone_ssm_higher_rank_dynamics_experiments.ipynb``` |


### Running Clean-Label Poisoning Experiments (Section 4.2)

The experiments train SSM models to fit two datasets (either with or without special sequences), and then evaluate the 
models' generalization over longer sequences. Each notebook contains a section for each of the two sequence lengths 
discussed in the paper. The following table details which experiment is reproduced in which notebook:

| Experiment                                        | Notebook                                             |
|---------------------------------------------------|------------------------------------------------------|
| Standalone SSM per Theorem 1 (Tables 1 and 2)     | ```standalone_ssm_theory_poison_experiments.ipynb``` |
| Standalone SSM beyond Theorem 1 (Tables 1 and 2)  | ```standalone_ssm_poison_experiments.ipynb```        |
| SSM in non-linear neural network (Tables 1 and 2) | ```ssm_in_nn_poison_experiments.ipynb```             |


## Citation

For citing the paper you can use:

```
@article{slutzky2024implicit,
  title={The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels},
  author={Slutzky, Yonatan and Alexander, Yotam and Razin, Noam and Cohen, Nadav},
  journal={arXiv preprint arXiv:2410.10473},
  year={2024}
}
```