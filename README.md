# The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels
Official implementation for the experiments in [The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](https://www.arxiv.org/abs/2410.10473), based on the [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/) and [SciPy](https://scipy.org/) libraries.

## Installing Requirements

Tested with Python 3.9. The ```requirements.txt``` file includes the required dependencies, which can be installed via:

```
pip install -r requirements.txt
```

## Experiments

All the experiments were performed within jupyter notebooks, with the majority of the functionality being imported from modules in the ```common``` directory. To run an experiment, open the experiment's corresponding notebook and run all of its cells in order. The experiments were carried out using a single Nvidia RTX 2080 Ti GPU. All notebooks (except ```standalone_ssm_theory_poison_experiments.ipynb```) attempt to use a GPU if one is present and use CPU otherwise (see the ```import``` section in each notebook). 

### Running Dynamical Characterization Experiments (Sections 4.1 and G.1)

The experiments train SSM models to fit two datasets (either with or without special sequences). Then, plots are drawn for the evolution of the following quantities:
* The absolute largest diagonal entries of the learned state transition matrices $A$.
* The effective rank of the learned state transition matrices $A$.
* The coefficient $\gamma^{(0)}(\cdot)$ in the expression of the derivative (see Proposition 1 of Section 3.1).

Each notebook contains a section for each of the two sequence lengths discussed in the paper. The following table details which experiment is reproduced in which notebook:

| Experiment                                               | Notebook                                              |
|----------------------------------------------------------|-------------------------------------------------------|
| Standalone SSM (Figures 2, 3, 7 and 8)                   | ```standalone_ssm_dynamics_experiments.ipynb```       |
| SSM in non-linear neural network (Figures 2, 3, 7 and 8) | ```ssm_in_nn_dynamics_experiments.ipynb```            |
| Standalone SSM dimension 2 (Figure 4)                    | ```standalone_ssm_dim=2_dynamics_experiments.ipynb``` |
| Standalone SSM dimension 3 (Figure 5)                    | ```standalone_ssm_dim=3_dynamics_experiments.ipynb``` |
| Standalone SSM dimension 4 (Figure 6)                    | ```standalone_ssm_dim=4_dynamics_experiments.ipynb``` |


### Running Clean-Label Poisoning Experiments (Sections 4.2 and G.2)

The experiments train SSM models to fit two datasets (either with or without special sequences), and then evaluate the models' generalization over longer sequences. Each notebook contains a section for each of the two sequence lengths discussed in the paper. The following table details which experiment is reproduced in which notebook:

| Experiment                                        | Notebook                                             |
|---------------------------------------------------|------------------------------------------------------|
| Standalone SSM per Theorem 1 (Tables 1 and 3)     | ```standalone_ssm_theory_poison_experiments.ipynb``` |
| Standalone SSM beyond Theorem 1 (Tables 1 and 3)  | ```standalone_ssm_poison_experiments.ipynb```        |
| SSM in non-linear neural network (Tables 1 and 3) | ```ssm_in_nn_poison_experiments.ipynb```             |


### Running Real-World Experiments (Section 4.2)

See the ```README.md``` file in the ```real-world experiments``` directory.

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
