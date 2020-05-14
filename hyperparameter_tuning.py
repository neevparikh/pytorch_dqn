import sys
from cluster_script import run
""" Tuning hyperparamters """

SEEDS_PER_RUN = 3
MODEL_BASE = "./saved_models"
OUTPUT_BASE = "./reward_log"

# Program args
default_args = [
    "--env", "PongNoFrameskip-v4",
    "--model-type", "cnn",
    "--batchsize", "32",
    "--replay-buffer-size", "1e6",
    "--warmup-period", "5e4",
    "--max-steps", "5e6",
    "--reward-clip", "1",
    "--epsilon-decay", "1.5e6",
    "--epsilon-decay-end", "0.05",
    "--test-policy-steps", "2.5e5",
]

# Values to tune
tuning_values = {
    "--target-moving-average": ["2e-2", "5e-3", "2e-3"],
    "--lr": ["5e-5", "2e-5", "7e-6"],
}


if __name__ == "__main__": 
    if sys.argv[0] != 1:
        raise RuntimeError(f"Need only one argument, path to venv. Got {sys.argv[0]}")
    ENV_PATH = sys.argv[1]
    seed = 0
    # Cluster args
    cluster_args = [
        "--jobtype", "gpu",
        "--env", ENV_PATH,
        "--nresources", "1",
        "--duration", "vlong",
    ]
    for arg, value_range in tuning_values:
        for value in value_range:
            for _ in range(SEEDS_PER_RUN):
                run_args = default_args + [arg, value]
                run_tag = f"{arg}_{value}"
                run_args += ["--uuid", run_tag]
                run_args += ["--seed", seed]
                run_args += ["--model-path", f"{MODEL_BASE}/{arg}/{value}"]
                run_args += ["--output-path", f"{OUTPUT_BASE}/{arg}/{value}"]
                cmd = "python train.py " + ' '.join(run_args)
                cluster_args += ["--command", cmd]
                cluster_args += ["--jobname", run_tag]
                run(custom_args=cluster_args)
                seed += 1
