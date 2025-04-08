# [Dual Focal Loss for Calibration]([url](https://arxiv.org/abs/2305.13665))

You can find Dual Focal Loss implementation in `dual_focal_loss.py`


## Usage
```python
from dual_focal_loss import DualFocalLoss

criterion = DualFocalLoss()

output = model(data)
loss = criterion(logits, targets)

```

# Hyperparameter $\gamma$

| Dataset     | Model        | $\gamma$ |
|-------------|--------------|-------|
| cifar100    | resnet50     | 5     |
| cifar100    | resnet110    | 6.1   |
| cifar100    | wide_resnet  | 3.9   |
| cifar100    | densenet121  | 3.4   |
|-------------|--------------|-------|
| cifar10     | resnet50     | 5     |
| cifar10     | resnet110    | 4.5   |
| cifar10     | wide_resnet  | 2.6   |
| cifar10     | densenet121  | 5     |
|-------------|--------------|-------|
| tiny_image  | resnet50  | 2.3   |

## Citation
If you find this repo useful, please cite our paper.
```
@inproceedings{tao2023dual,
  title={Dual focal loss for calibration},
  author={Tao, Linwei and Dong, Minjing and Xu, Chang},
  booktitle={International Conference on Machine Learning},
  pages={33833--33849},
  year={2023},
  organization={PMLR}
}
```
