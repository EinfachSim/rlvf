"""
Print the size of the b/d vectors each HN produces, to set the random
condition's --random_b_scale / --random_d_scale. CPU only.

Run from the bench folder:
  python hn_scales.py --hn_module rlvf.model \
      --ckpt pathwise=/path/pathwise.pt warmstart=/path/warm.pt coldstart=/path/cold.pt
"""
import argparse

import torch

from aita_eval import load_hypernetwork
from rlvf.data import ValueProfileSampler

ap = argparse.ArgumentParser()
ap.add_argument("--hn_module", default="rlvf.model")
ap.add_argument("--ckpt", nargs="+", required=True, help="name=path ...")
args = ap.parse_args()

profiles = torch.tensor(ValueProfileSampler(rng=42).sample_batch(64),
                        dtype=torch.float32)

for item in args.ckpt:
    name, path = item.split("=", 1)
    act, A, B = load_hypernetwork(path, "cpu", args.hn_module)
    b, d = act(profiles)
    b_all = torch.cat([v.flatten() for v in b.values()])
    print(f"{name:10s} b: mean={b_all.mean():+.4f} std={b_all.std():.4f} |b|={b_all.abs().mean():.4f}   "
          f"d: mean={d.mean():+.4f} std={d.std():.4f} |d|={d.abs().mean():.4f}")

print("\nmatch the random condition with: --random_b_scale <b std> "
      "--random_d_mean <d mean> --random_d_scale <d std>")