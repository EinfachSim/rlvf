from rlvf.bench_v2.env import BenchEnv
import ray
import time
from pathlib import Path
import pickle

conditions = {
    "prompted": {}, #doesnt need any kwargs
    "pathwise": {
        "hn_path": "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/checkpoints_pathwise_v6/hn_step_00570.pt",
        "name": "pathwise"
    },
    "warmstart": {
        "hn_path": "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/checkpoints_sharpening_v8/hn_step_00155.pt",
        "name": "warmstart"
    },
    "base": {}
}

ray.init()

aita_results = BenchEnv(conditions, num_workers=3, benchmark="aita", eval_size=128, include_s2b=True, s2b_only=True).step_batch()


stamp = time.strftime("%Y%m%d-%H%M%S")
out = "/home/s03skoeh_hpc/MA/results"
out = Path(out)
out.mkdir(parents=True, exist_ok=True)

with open(out / f"raw_{stamp}.pkl", "wb") as f:
    pickle.dump(aita_results, f)

print(f"[AITA] Done. Saved results to {out / f'raw_{stamp}.pkl'}")