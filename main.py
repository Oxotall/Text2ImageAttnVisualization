"""CLI entry point: generate an image and save its attention visualizations.

Example:
    python main.py --prompt "a cat sitting on a red sofa" --steps 30 --out runs/cat
"""

from __future__ import annotations

import argparse

from attnviz import Config, DeviceResolver, Generator, PipelineLoader
from attnviz.credentials import load_hf_token
from attnviz.figure_saver import FigureSaver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize SD 1.5 attention maps.")
    p.add_argument("--prompt", required=True, help="Text prompt to generate.")
    p.add_argument("--out", default="runs/latest", help="Output directory.")
    p.add_argument("--steps", type=int, default=30, help="Denoising steps.")
    p.add_argument("--size", type=int, default=512, help="Image size (px).")
    p.add_argument("--guidance", type=float, default=7.5, help="CFG scale.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--device", default=None, help="cuda|mps|cpu (auto if unset).")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                   help="HuggingFace model id.")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        model_id=args.model,
        device=args.device,
        image_size=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    load_hf_token()  # optional — reads .env if present, else no-op
    config = build_config(args)

    resolver = DeviceResolver(config.device, config.dtype)
    print(f"Using {resolver}")

    loader = PipelineLoader(config, resolver)
    print(f"Loading model {config.model_id} ...")
    loader.load()

    generator = Generator(config, loader)
    print(f"Generating: {args.prompt!r}")
    result = generator.generate(args.prompt)

    saver = FigureSaver(args.out)
    paths = saver.save_all(result)
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
