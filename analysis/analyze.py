"""Аналитика SDXL: параметры, FLOPs по блокам, память. Считается на meta-устройстве (без весов)."""
import json, torch
from torch.utils.flop_counter import FlopCounterMode
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

UNET = dict(
    sample_size=128, in_channels=4, out_channels=4,
    down_block_types=["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    block_out_channels=[320, 640, 1280], layers_per_block=2,
    transformer_layers_per_block=[1, 2, 10], attention_head_dim=[5, 10, 20],
    cross_attention_dim=2048, use_linear_projection=True,
    addition_embed_type="text_time", addition_time_embed_dim=256,
    projection_class_embeddings_input_dim=2816,
)
VAE = dict(in_channels=3, out_channels=3, latent_channels=4,
           down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4,
           block_out_channels=[128, 256, 512, 512], layers_per_block=2, sample_size=1024,
           scaling_factor=0.13025)
TE1 = CLIPTextConfig(hidden_size=768, intermediate_size=3072, num_attention_heads=12,
                     num_hidden_layers=12, projection_dim=768, hidden_act="quick_gelu")
TE2 = CLIPTextConfig(hidden_size=1280, intermediate_size=5120, num_attention_heads=20,
                     num_hidden_layers=32, projection_dim=1280, hidden_act="gelu")

def nparams(m):
    return sum(p.numel() for p in m.parameters())

def by_prefix(counter, depth):
    out = {}
    for k, v in counter.get_flop_counts().items():
        parts = k.split(".")
        if len(parts) == depth:
            out[k] = sum(v.values())
    return out

res = {}
with torch.device("meta"):
    unet = UNet2DConditionModel(**UNET)
    vae = AutoencoderKL(**VAE)
    te1 = CLIPTextModel(TE1)
    te2 = CLIPTextModelWithProjection(TE2)
res["params"] = {"unet": nparams(unet), "vae": nparams(vae), "vae_decoder": nparams(vae.decoder),
                 "text_encoder_1": nparams(te1), "text_encoder_2": nparams(te2)}

# FLOPs UNet на один вызов (batch=1) для разных разрешений
res["unet_flops"] = {}
for side in [512, 768, 1024]:
    lat = side // 8
    x = torch.randn(1, 4, lat, lat, device="meta")
    t = torch.tensor([500], device="meta")
    ctx = torch.randn(1, 77, 2048, device="meta")
    cond = {"text_embeds": torch.randn(1, 1280, device="meta"), "time_ids": torch.randn(1, 6, device="meta")}
    fc = FlopCounterMode(display=False)
    with fc:
        unet(x, t, encoder_hidden_states=ctx, added_cond_kwargs=cond)
    total = fc.get_total_flops()
    entry = {"total": total}
    # разбивка по операциям
    ops = {}
    for mod, d in fc.get_flop_counts().items():
        if mod == "Global":
            for op, v in d.items():
                ops[str(op)] = v
    entry["by_op"] = ops
    # разбивка по верхним блокам
    blocks = {}
    for mod, d in fc.get_flop_counts().items():
        p = mod.split(".")
        if len(p) == 3 and p[1] in ("down_blocks", "up_blocks"):
            blocks[f"{p[1]}.{p[2]}"] = sum(d.values())
        if len(p) == 2 and p[1] == "mid_block":
            blocks["mid_block"] = sum(d.values())
    entry["by_block"] = blocks
    # attention: sdpa vs остальное в трансформерах
    res["unet_flops"][side] = entry

# VAE decoder FLOPs на 1024
fc = FlopCounterMode(display=False)
with fc:
    vae.decode(torch.randn(1, 4, 128, 128, device="meta"))
res["vae_decode_flops_1024"] = fc.get_total_flops()

# text encoders
fc = FlopCounterMode(display=False)
ids = torch.zeros(1, 77, dtype=torch.long, device="meta")
with fc:
    te1(ids); te2(ids)
res["text_enc_flops"] = fc.get_total_flops()

print(json.dumps(res, indent=1, default=str))
json.dump(res, open("analysis/analysis.json", "w"), indent=1, default=str)
