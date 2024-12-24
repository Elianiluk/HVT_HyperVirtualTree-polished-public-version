# HVT: HyperVirtualTree
This is the official implementation of HVT. We utilized a hypernetwork to generate tree branches that are differentiable only by the dynamically generated input to the hypernetwork.

![hyper_virtual_tree_concept.png](images%2Fhyper_virtual_tree_concept.png)

**left image**: normal decision trees.

**right image**: our proposed method, the branches virtual in the sense that the only diffrence between them is the input to the hypernetwork that creates the filter weights. this input generated dynamically at run-time.

## Prerequisites
- Python 3.7+
- PyTorch 1.9+
- CUDA 11.1 (in case you are using a GPU)

Other than that, create a virtual environment and run `pip install -r requirements.txt`.

We ran the code only on linux, it will not necessarily work on other operating systems.

## Training
Note: it is advised to set PYTHONPATH to the root directory
```
# For training baseline models, make sure the relevant trainer is set in baseline_trainers.py
# The run: 
python trainers/baseline_trainers.py -d <dataset name>
# (The dataset name is the only positional argument)

# For training a HVT model, make sure the correct HVT model is set and run: 
python trainers/HVT_trainers.py -d <dataset name> --beta <value>
```
### Optional arguments (all trainers):
```
  -h, --help            show this help message and exit
  --dataset_name {CIFAR10,CIFAR100,FashionMNIST}, -d {CIFAR10,CIFAR100,FashionMNIST}
                        which dataset to train
  --exp_name EXP_NAME, -e EXP_NAME
                        Name of the experiment (positional when using W&B)
  --weights_path WEIGHTS_PATH, -w WEIGHTS_PATH
                        Path to the model's weights file
  --batch_size BATCH_SIZE
                        Train batch size
  --test_batch_size TEST_BATCH_SIZE
                        Test batch size
  --num_epochs NUM_EPOCHS
                        Number of training epochs
  --num_workers NUM_WORKERS
                        Number of dataloader workers
  --include_top5        Whether to log top5 accuracy data
  --use_wandb           Track run with Weights and Biases
  --log_images          Log images to wandb (only works if use_wandb=True
  --no_save             Do not save checkpoints
  --learning_rate LEARNING_RATE
                        Optimizer initial learning rate
  --do_early_stopping   Enable early stopping
  --augment             Perform data augmentation
  --use_validation      Use validation set
  --early_stopping_params EARLY_STOPPING_PARAMS
                        JSON string with the EarlyStopping config dict
  --lr_change_factor LR_CHANGE_FACTOR
                        LR change factor (for the LR scheduler)
  --num_lr_changes NUM_LR_CHANGES
                        The number of LR changes allowed for the LR scheduler
  --resume_run_id RESUME_RUN_ID
                        wandb run-id for resuming crashed runs (warning: this
                        was not used thoroughly; use with caution)
  
```
### Additional optional arguments (HVT trainers only):
```
 --beta BETA           weight for the sigma loss
 --always_binarize     do not use non-binary values in the binarization layer
                       (i.e., perform only hard routing)
 --include_entropy     include printing branch entropy
```
#### Weights & Biases
The training framework fully supports logging metrics to the wonderful [Weights & Biases](www.wandb.ai) ML-Ops framework. Simply add `--use_wandb` when training; 
Accuracy, loss and learning rates are logged automatically.

## Issues
If you encounter any running issues, feel free to open an issue. We'll try to look into it and fix as needed.

## Citation
If you find this code useful, please cite our paper:
```
@InProceedings{
    author    = 
    title     = 
    booktitle = 
    month     = 
    year      = 
    pages     = 
}
```

## Acknowledgements
this code based on the official code of ["DecisioNet: A Binary-Tree Structured Neural Network"] by Noam Gottlieb and Michael Werman.
