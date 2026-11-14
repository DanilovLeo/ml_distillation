"""Замеры SDXL на GPU: латентность по этапам и пиковая память для конфигураций из README (раздел 5).
Следующий этап: в ДЗ №1 скрипт не запускался, гипотезы сформулированы аналитически.

Примеры:
    python benchmark/benchmark.py --config baseline
    python benchmark/benchmark.py --config all --runs 3
"""
import argparse, csv, os, time
import torch
from diffusers import (StableDiffusionXLPipeline, AutoencoderKL,
                       DPMSolverMultistepScheduler, LCMScheduler)

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"   # VAE без переполнения в fp16
LCM_LORA = "latent-consistency/lcm-lora-sdxl"
PROMPT = "a photo of an astronaut riding a horse on mars, highly detailed"

CONFIGS = {
    # name: (steps, guidance, resolution, extra)
    "baseline":   dict(steps=30, cfg=5.0, res=1024),
    "dpm20":      dict(steps=20, cfg=5.0, res=1024, sched="dpm"),
    "lcm4":       dict(steps=4,  cfg=1.0, res=1024, lcm=True),
    "deepcache":  dict(steps=30, cfg=5.0, res=1024, deepcache=True),
    "tome":       dict(steps=30, cfg=5.0, res=1024, tome=0.5),
    "res768":     dict(steps=30, cfg=5.0, res=768),
    "vae_tiling": dict(steps=30, cfg=5.0, res=1024, vae_tiling=True),
    "compile":    dict(steps=30, cfg=5.0, res=1024, compile=True),
    "offload":    dict(steps=30, cfg=5.0, res=1024, offload=True),
}


def build(cfg):
    vae = AutoencoderKL.from_pretrained(VAE_FIX, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    if cfg.get("sched") == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if cfg.get("lcm"):
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(LCM_LORA)
        pipe.fuse_lora()
    if cfg.get("offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if cfg.get("vae_tiling"):
        pipe.enable_vae_tiling()
    if cfg.get("tome"):
        import tomesd  # pip install tomesd
        tomesd.apply_patch(pipe, ratio=cfg["tome"])
    helper = None
    if cfg.get("deepcache"):
        from DeepCache import DeepCacheSDHelper  # pip install DeepCache
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
    if cfg.get("compile"):
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
    return pipe


def timed_run(pipe, cfg, seed):
    """Время UNet-цикла и VAE отдельно: латент получаем через output_type='latent'."""
    g = torch.Generator("cuda").manual_seed(seed)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    lat = pipe(PROMPT, num_inference_steps=cfg["steps"], guidance_scale=cfg["cfg"],
               height=cfg["res"], width=cfg["res"], generator=g, output_type="latent").images
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.no_grad():
        vae = pipe.vae
        lat = lat.to("cuda", torch.float16) / vae.config.scaling_factor
        img = vae.decode(lat).sample
    torch.cuda.synchronize(); t2 = time.perf_counter()
    nan = bool(torch.isnan(img).any())
    img = pipe.image_processor.postprocess(img, output_type="pil")[0]
    return t1 - t0, t2 - t1, nan, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline", help="имя конфига или 'all'")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()
    names = list(CONFIGS) if args.config == "all" else [args.config]
    os.makedirs("images", exist_ok=True)
    print(torch.cuda.get_device_name(0), torch.__version__)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["config", "steps", "res", "unet_loop_s", "vae_s", "total_s",
                        "peak_mem_gb", "nan"])
        for name in names:
            cfg = CONFIGS[name]
            pipe = None
            try:
                pipe = build(cfg)
                timed_run(pipe, cfg, seed=0)  # прогрев (и компиляция для 'compile')
                torch.cuda.reset_peak_memory_stats()
                res = [timed_run(pipe, cfg, seed=i) for i in range(args.runs)]
                u = sum(r[0] for r in res) / len(res)
                v = sum(r[1] for r in res) / len(res)
                mem = torch.cuda.max_memory_allocated() / 1e9
                nan = any(r[2] for r in res)
                res[0][3].save(f"images/{name}.png")
                w.writerow([name, cfg["steps"], cfg["res"], f"{u:.2f}", f"{v:.2f}",
                            f"{u + v:.2f}", f"{mem:.2f}", nan])
                print(f"{name:11s} unet {u:6.2f}s  vae {v:5.2f}s  peak {mem:5.2f} GB  nan={nan}")
            except torch.cuda.OutOfMemoryError:
                w.writerow([name, cfg["steps"], cfg["res"], "", "", "", "OOM", ""])
                print(f"{name:11s} OOM")
            finally:
                f.flush()
                pipe = None
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
