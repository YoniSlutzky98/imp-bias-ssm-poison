# Real World Experiments

Code for reproducing our real-world experiments on CIFAR-10 (Table 2) is included in this sub-repository. 
We used the [gradient-matching poisoning method implementation](https://github.com/JonasGeiping/poisoning-gradient-matching) of Jonas Geiping et. al', making adaptions to fit our needs:
- added support for the S4, Mamba2 and LRU models via adaptations of the [minimalist S4](https://github.com/state-spaces/s4/tree/main/models/s4), [mamba2-minimal](https://github.com/tommyip/mamba2-minimal) and [minimal LRU](https://github.com/tommyip/mamba2-minimal) implementations, respectively.
- adjusted code such that poisonous examples are included in both their clean and poisoned forms
- included a smarter allocation of poison to individual targets
- introduced a regularization that encourages the last elements of an injected noise sequence to be relatively large

The base shell command for running each of the experiments is
```
python brew_poison.py --retrain_from_init --save numpy --continue_training_to_loss --duplicate_poison_training
```
The table below gives the additional arguments required to reproduce each of the rows in Table 2 (3 rows per experiment).

| Row                                     | Command                                                       |
|-----------------------------------------|---------------------------------------------------------------|
| S4  | ```--net S4 --optimization s4 --poisonkey 17 --modelkey 27``` |
| S4  | ```--net S4 --optimization s4 --poisonkey 18 --modelkey 28``` |
| S4  | ```--net S4 --optimization s4 --poisonkey 19 --modelkey 29``` |
| Mamba2 | ```--net Mamba2 --optimization mamba2 --poisonkey 11 --modelkey 21```  |
| Mamba2 | ```--net Mamba2 --optimization mamba2 --poisonkey 12 --modelkey 22```  |
| Mamba2 | ```--net Mamba2 --optimization mamba2 --poisonkey 16 --modelkey 26```  |
| LRU | ```--net LRU --optimization lru --poisonkey 13 --modelkey 23``` |
| LRU | ```--net LRU --optimization lru --poisonkey 14 --modelkey 24``` |
| LRU | ```--net LRU --optimization lru --poisonkey 15 --modelkey 25``` |

By default the code runs on all available GPUs. Adjusting this behaviour can be done via the ```--gpus``` and ```--max_gpus``` options. 


After an experiment terminates, a ```.csv``` table reporting the results is created in ```./tables```. The table below details the relevant columns and their descriptions.

| Column                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| ```train_acc_clean```   | training accuracy of the model trained without poisoning                    |
| ```val_acc_clean```     | test accuracy of the model trained without poisoning                        |
| ```train_loss_clean```   | training loss of the model trained without poisoning                        |
| ```val_loss_clean```     | test loss of the model trained without poisoning                            |
| ```train_acc_rerun```   | training accuracy of the model trained with poisoning                       |
| ```val_acc_rerun```     | test accuracy of the model trained with poisoning                           |
| ```train_loss_rerun```   | training loss of the model trained with poisoning                           |
| ```val_loss_rerun```     | test loss of the model trained with poisoning                               |
| ```last_row_norm_ratio```     | average relative size of the last elements in the generated poisonous noise |
