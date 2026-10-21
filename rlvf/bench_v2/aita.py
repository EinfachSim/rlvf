from rlvf.bench_v2.env import BenchEnv
import ray

conditions = {
    #"prompted": {},
    "pathwise": {
        "hn_path": "/lustre/mlnvme/data/s03skoeh_hpc-rlvf/checkpoints_pathwise_v6/hn_step_00350.pt",
        "name": "pathwise"
    }
}

ray.init()

BenchEnv(conditions, num_workers=1, benchmark="aita", eval_size=2).step_batch()