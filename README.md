# [Dual Focal Loss for Calibration]([url](https://arxiv.org/abs/2305.13665))

You can find Dual Focal Loss implementation in `Losses/dual_focal_loss.py`


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
@misc{tao2023dual,
      title={Dual Focal Loss for Calibration}, 
      author={Linwei Tao and Minjing Dong and Chang Xu},
      year={2023},
      eprint={2305.13665},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
