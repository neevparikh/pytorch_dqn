import sys
import os
import stat
import subprocess
from cluster_script import run as run_csgrid
from ccv_script import run as run_ccv

# Tuning hyperparameters

SEED_START = 0
SEEDS_PER_RUN = 3
MODEL_BASE = "./saved_models"
OUTPUT_BASE = "./reward_log"

# Program args
default_args = [
#    "--env", "PongNoFrameskip-v4",
#    "--env", "SpaceInvadersNoFrameskip-v4",
#    "--env", "SeaquestNoFrameskip-v4",
#    "--env", "BreakoutNoFrameskip-v4",
#    "--env", "QbertNoFrameskip-v4",
#    "--env", "RiverraidNoFrameskip-v4",
    "--model-type", "cnn",
#    "--gpu",
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
python hyperparameter_tuning.py /path/to/env/ [ccv | csgrid | no_grid]""")
    ENV_PATH = sys.argv[1]
    grid_type = sys.argv[2]
    seed = SEED_START
    # Cluster args
    if grid_type == "ccv":
        cluster_args = [
            "--cpus", "4",
            "--mem", "10",
            "--env", ENV_PATH,
            "--duration", "vlong",
        ]
    elif grid_type == "csgrid":
        cluster_args = [
            "--jobtype", "cpu",
            "--mem", "10",
            "--nresources", "4",
            "--env", ENV_PATH,
            "--duration", "vlong",
        ]
    elif grid_type == "no_grid":
        pass                                           
    else:
        raise RuntimeError("""Usage:
python hyperparameter_tuning.py /path/to/env/ [ccv | csgrid | no_grid]""")

    for _ in range(SEEDS_PER_RUN):
        for i, (arg1, value_range1) in enumerate(tuning_values.items()):
            for j, (arg2, value_range2) in enumerate(tuning_values.items()):
                if j > i:
                    continue
                if arg1 == arg2:
                    continue
                for value1 in value_range1:
                    for value2 in value_range2:
                        run_args = default_args + [arg1, value1] + [arg2, value2]
                        clean_arg_name1 = arg1.strip('-').replace('-', '_')
                        clean_arg_name2 = arg2.strip('-').replace('-', '_')
                        run_tag = f"{clean_arg_name1}_{value1}_{clean_arg_name2}_{value2}" + "_cpu"
                        run_args += ["--uuid", run_tag]
                        run_args += ["--seed", str(seed)]
                        run_args += ["--model-path", f"{MODEL_BASE}/{clean_arg_name1}_{clean_arg_name2}/{value1}_{value2}"]
                        run_args += ["--output-path", f"{OUTPUT_BASE}/{clean_arg_name1}_{clean_arg_name2}/{value1}_{value2}"]
                        cmd = "python train.py " + ' '.join(run_args)
                        jobname = f"{default_args[1].replace('-', '_')}_{run_tag.replace('-','_')}_seed_{str(seed)}"
                        if grid_type != "no_grid":
                            cmd = "unbuffer " + cmd
                            cluster_args += ["--command", cmd]
                            cluster_args += ["--jobname", jobname]
                        if grid_type == "ccv":
                            run_ccv(custom_args=cluster_args)
                        elif grid_type == "csgrid":
                            run_csgrid(custom_args=cluster_args)
                        elif grid_type == "no_grid":
                            os.makedirs("./jobs/logs", exist_ok=True)
                            os.makedirs("./jobs/scripts", exist_ok=True)
                            print(cmd)
                            with open(f"./jobs/scripts/{jobname}", "w+") as sc:
                                sc.write(f"""
#!/usr/bin/env bash
{cmd}""")
                            os.chmod(f"./jobs/scripts/{jobname}", stat.S_IRWXU)
                            script_cmd = f"CUDA_VISIBLE_DEVICES={seed % 4} ./jobs/scripts/{jobname}"
                            subprocess.Popen(script_cmd, shell=True)
                        else:
                            raise RuntimeError("""Usage:
python hyperparameter_tuning.py /path/to/env/ [ccv | csgrid]""")

                        seed += 1
