from rlvf.bench_v2.env import BenchEnv
import ray
import time
from pathlib import Path
import pickle

conditions = {
    "prompted": {}, #doesnt need any kwargs
    "pathwise": {
        "hn_path": "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/checkpoints_pathwise_v6/hn_step_00350.pt",
        "name": "pathwise"
    }
}

ray.init()

aita_results = BenchEnv(conditions, num_workers=1, benchmark="aita", eval_size=2).step_batch()


stamp = time.strftime("%Y%m%d-%H%M%S")
out = "home/s03skoeh_hpc/MA/results"
out = Path(out)
out.mkdir(parents=True, exist_ok=True)

with open(out / f"raw_{stamp}.pkl", "wb") as f:
    pickle.dump(aita_results, f)

print(f"[AITA] Done. Saved results to {out / f'raw_{stamp}.pkl'}")