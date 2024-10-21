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
@article{tao2023calibrating,
  title={Calibrating a deep neural network with its predecessors},
  author={Tao, Linwei and Dong, Minjing and Liu, Daochang and Sun, Changming and Xu, Chang},
  journal={arXiv preprint arXiv:2302.06245},
  year={2023}
}
```
