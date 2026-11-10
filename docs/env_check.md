# Environment check — CFP-9 (updated after R2-0)

| Item | Value |
|---|---|
| Python | 3.10.19, project-local `.venv` |
| GPU | RTX 5060 Laptop, 8 GB, sm_120, driver 592.27 |
| torch | **2.8.0+cu128** (cu121 cannot run sm_120) |
| torchvision | 0.23.0+cu128 |
| timm | 0.9.2 (1.x breaks `models_vit.py`) |
| numpy | 1.26.4 (unchanged) |
| Verification | `torch.cuda.is_available() == True`, `get_device_capability() == (12, 0)`, `pip check` clean |

Installed for later stages: huggingface_hub, matplotlib, seaborn, cleanvision, cleanlab, dvc,
pandera, pytest, imagehash, opencv-python, scikit-learn, scipy.

Checkpoint: full training checkpoint (`model/optimizer/epoch/scaler/args`), 398 tensors, no
classifier head, `mae_vit_large_patch16`, 224 px, epoch 800. `fc_norm` is absent from the
checkpoint; `norm.*` is remapped onto `fc_norm.*` so frozen features use the pretrained
LayerNorm (`missing=0, unexpected=106` = MAE decoder/mask tokens).

Network note: install PyTorch from the official cu128 wheel index. If the network requires a proxy, configure it in the local shell; do not commit proxy credentials.
