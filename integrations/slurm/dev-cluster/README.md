# SLURM Dev Cluster

This is a development cluster for working on the SLURM Juice Plugin.

## Starting the cluster

`docker compose up -d`

Connect to the slurmctld instance to run SLURM commands:

`docker compose exec slurmctld /bin/bash`

## Running SLURM commands

SLURM has two main commands `srun` and `sbatch`.

 * srun will run your command in interactive mode, with it streaming output from a compute node elsewhere in the cluster. srun commands will fail immediately if they cannot be scheduled

 * sbatch will queue a job. The job may run immediately or later when resources are available. All output from the job is written to slurm-<jobid>.out in the directory you execute the job from. This is why its important to execute the job in a directory that has shared storage across all nodes. (In this dev cluster /data)

Examples:

| Command | Description |
|---------|-------------|
| srun hostname -A | Interactively runs "hostname -A" on a compute node. You should see "compute1" output on the terminal even though you're on slurmctld |
| sbatch --wrap="hostname -A" | Queues a job to "hostname -A" on a compute node. You should see "compute1" written to a file on disk. (Make sure you're in /data when you run it!) |
| srun --juice-pool=test hostname -A | Interactively "hostname -A" on a compute node after obtaining and connecting to a Juice Session |
| sbatch --juice-pool=test --wrap="hostname -A" | Queues a job to run "hostname -A" on a compute node after obtaining and connecting to a Juice Session |

## Developing the Spank Plugin

The source files are read from your host machine. So you can make changes while the cluster is running, but then you need to compile it in the container. To do this, exec into the slurmctld container and run `make` in the /spank directory. Slurm reloads the plugin for every execution so your changes will be visible immedately.



## Developing the Prolog/Epilog Hooks

The scripts are loaded from your host machine, so you can change the scripts at any time and then run a new job