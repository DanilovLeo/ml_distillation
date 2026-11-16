"""Параметры и FLOPs DETR-R50 по частям. Веса случайные, на FLOPs это не влияет."""
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import DetrConfig, DetrForObjectDetection, ResNetConfig

cfg = DetrConfig(use_timm_backbone=False, use_pretrained_backbone=False,
                 backbone_config=ResNetConfig(out_features=["stage4"]))
model = DetrForObjectDetection(cfg).eval()

parts = {"backbone": model.model.backbone, "encoder": model.model.encoder,
         "decoder": model.model.decoder}
for name, m in parts.items():
    print(f"{name}: {sum(p.numel() for p in m.parameters()) / 1e6:.1f} M")
print(f"всего: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M параметров")

for h, w in [(800, 1333), (512, 853)]:
    x = torch.randn(1, 3, h, w)
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        model(pixel_values=x)
    counts = fc.get_flop_counts()
    total = fc.get_total_flops()
    print(f"\nвход {h}x{w}: {total / 1e9:.1f} GFLOPs, токенов {(h // 32 + (h % 32 > 0)) * (w // 32 + (w % 32 > 0))}")
    for name in parts:
        key = next(k for k in counts if k.endswith("." + name) and k.count(".") == 2)
        f = sum(counts[key].values())
        print(f"  {name}: {f / 1e9:.1f} GFLOPs ({100 * f / total:.0f}%)")
