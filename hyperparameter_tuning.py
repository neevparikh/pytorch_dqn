import sys
from cluster_script import run as run_csgrid
from ccv_script import run as run_ccv
""" Tuning hyperparameters """

SEEDS_PER_RUN = 3
MODEL_BASE = "./saved_models"
OUTPUT_BASE = "./reward_log"

# Program args
default_args = [
    "--env", "BreakoutNoFrameskip-v4",
    "--model-type", "cnn",
    "--gpu",
    "--batchsize", "32",
    "--replay-buffer-size", "1e6",
    "--warmup-period", "5e4",
    "--max-steps", "5e6",
    "--reward-clip", "1",
    "--epsilon-decay", "1.5e6",
    "--epsilon-decay-end", "0.05",
    "--test-policy-steps", "1e5",
]

# Values to tune
tuning_values = {
    "--target-moving-average": ["2e-2", "5e-3", "2e-3"],
    "--lr": ["5e-5", "2e-5", "7e-6"],
}


if __name__ == "__main__": 
    if len(sys.argv) != 3:
        raise RuntimeError("""Usage:
python hyperparameter_tuning.py /path/to/env/ [ccv | csgrid]""")
    ENV_PATH = sys.argv[1]
    grid_type = sys.argv[2]
    seed = 0
    # Cluster args
    if grid_type == "ccv":
        cluster_args = [
            "--cpus", "4",
            "--gpus", "1",
            "--mem", "12",
            "--env", ENV_PATH,
            "--duration", "vlong",
        ]
    elif grid_type == "csgrid":
        cluster_args = [
            "--jobtype", "gpu",
            "--env", ENV_PATH,
            "--duration", "vlong",
        ]
    else:
        raise RuntimeError("""Usage:
python hyperparameter_tuning.py /path/to/env/ [ccv | csgrid]""")

    for i, (arg1, value_range1) in enumerate(tuning_values.items()):
        for j, (arg2, value_range2) in enumerate(tuning_values.items()):
            if j > i:
                continue
            if arg1 == arg2:
                continue
            for value1 in value_range1:
                for value2 in value_range2:
                    for _ in range(SEEDS_PER_RUN):
                        run_args = default_args + [arg1, value1] + [arg2, value2]
                        clean_arg_name1 = arg1.strip('-').replace('-', '_')
                        clean_arg_name2 = arg2.strip('-').replace('-', '_')
                        run_tag = f"{clean_arg_name1}_{value1}_{clean_arg_name2}_{value2}"
                        run_args += ["--uuid", run_tag]
                        run_args += ["--seed", str(seed)]
                        run_args += ["--model-path", f"{MODEL_BASE}/{clean_arg_name1}_{clean_arg_name2}/{value1}_{value2}"]
                        run_args += ["--output-path", f"{OUTPUT_BASE}/{clean_arg_name1}_{clean_arg_name2}/{value1}_{value2}"]
                        cmd = "python train.py " + ' '.join(run_args)
                        cluster_args += ["--command", cmd]
                        cluster_args += ["--jobname", f"{run_tag.replace('-','_')}_{str(seed)}"]
                        if grid_type == "ccv":
                            run_ccv(custom_args=cluster_args)
                        elif grid_type == "csgrid":
                            run_csgrid(custom_args=cluster_args)
                        else:
                            raise RuntimeError("""Usage:
python hyperparameter_tuning.py /path/to/env/ [ccv | csgrid]""")

                        seed += 1
