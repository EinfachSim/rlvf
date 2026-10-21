from rlvf.bench_v2.env import BenchEnv
import ray

conditions = {
    "prompted": {},
    "pathwise": {
        "hn_path": "/lustre/mlnvme/data/s03skoeh_hpc/checkpoints_pathwise_v6",
        "name": "pathwise"
    }
}

ray.init()

BenchEnv(conditions, num_workers=2)