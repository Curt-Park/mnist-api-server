## How to train

It saves the model with the best accuracy from training.

```bash
$ python train.py --help
usage: train.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N] [--lr LR] [--gamma M] [--no-cuda] [--seed S] [--log-interval N] [--save-model]

PyTorch MNIST Example

optional arguments:
  -h, --help           show this help message and exit
  --batch-size N       input batch size for training (default: 64)
  --test-batch-size N  input batch size for testing (default: 1000)
  --epochs N           number of epochs to train (default: 14)
  --lr LR              learning rate (default: 1.0)
  --gamma M            Learning rate step gamma (default: 0.7)
  --no-cuda            disables CUDA training
  --seed S             random seed (default: 1)
  --log-interval N     how many batches to wait before logging training status
  --save-model         For Saving the current Model
```
