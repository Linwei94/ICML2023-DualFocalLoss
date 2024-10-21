# [Dual Focal Loss for Calibration]([url](https://arxiv.org/abs/2305.13665))

You can find Dual Focal Loss implementation in `dual_focal_loss.py`


## Usage
```python
from dual_focal_loss import DualFocalLoss

criterion = DualFocalLoss()

output = model(data)
loss = criterion(logits, targets)

```

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
