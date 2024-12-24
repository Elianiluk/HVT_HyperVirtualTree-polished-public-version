import subprocess

# Define a list of different parameter values
params = ["128"]

# Loop through the parameter values
for param in params:
    # Construct the command with the current parameter value
    command = [
        "/home/itamar/HyperDecisioNet/venv/bin/python",
        "/home/itamar/HyperDecisioNet/trainers/baseline_trainers.py",
        "-d", "CIFAR10",
        "-e", "hyper_ensemble_voting_scale_sweep",
        "--network_type", "hyper-ensemble-voting",
        "--augment",
        "--use_wandb",
        "--num_epochs", "300",
        "--scale_factor", param
    ]

    # Run the command
    subprocess.run(command)
