# Real World Experiments

Code for reproducing our real-world experiments on CIFAR-10 (Table 2) is included in this sub-repository. 
We used the [gradient-matching poisoning method implementation](https://github.com/JonasGeiping/poisoning-gradient-matching) of Jonas Geiping et. al', making slight adaptions to fit our needs:
- added support for the S4 model via the [minimalist S4 implementation](https://github.com/state-spaces/s4/tree/main/models/s4)
- adjusted code such that the clean training set is always of size 45000, and the poisoned examples are added to it (as opposed to poisoning existing training examples) 
- introduced a regularization that encourages the last elements of an injected noise sequence to be relatively large

The base shell command for running each of the experiments is
```
python brew_poison.py --net S4 --threatmodel random-subset
```
The table below gives the additional arguments required to reproduce each of the rows in Table 2.

| Row                                     | Command                                                       |
|-----------------------------------------|---------------------------------------------------------------|
| 10 test instances, 500 poison examples  | ```--poisonkey 15 --modelkey 25 --budget 0.01 --targets 10``` |
| 10 test instances, 2500 poison examples | ```--poisonkey 13 --modelkey 23 --budget 0.05 --targets 10``` |
| 10 test instances, 5000 poison examples | ```--poisonkey 11 --modelkey 21 --budget 0.1 --targets 10```  |
| 20 test instances, 500 poison examples  | ```--poisonkey 14 --modelkey 24 --budget 0.01 --targets 20``` |
| 20 test instances, 2500 poison examples | ```--poisonkey 12 --modelkey 22 --budget 0.05 --targets 20``` |
| 20 test instances, 5000 poison examples | ```--poisonkey 10 --modelkey 20 --budget 0.1 --targets 20```  |

After an experiment terminates, a ```.csv``` table reporting the results is created in ```./tables```. The table below details the relevant columns and their descriptions.

| Column                  | Description                                                            |
|-------------------------|------------------------------------------------------------------------|
| ```train_acc_clean```   | training accuracy of the model trained without poisoning               |
| ```val_acc_clean```     | validation accuracy of the model trained without poisoning             |
| ```target_acc_clean```  | accuracy on the target examples of the model trained without poisoning |
| ```train_acc_reinit```  | training accuracy of the model trained with poisoning                  |
| ```val_acc_reinit```    | validation accuracy of the model trained with poisoning                |
| ```target_acc_reinit``` | accuracy on the target examples of the model trained with poisoning    |
