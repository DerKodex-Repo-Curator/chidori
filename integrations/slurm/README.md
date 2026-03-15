# Slurm

This SLURM plugin simplifies the integration of jobs running in your SLURM cluster by matching them with Juice Agents running outside the cluster.

## Architecture

![SlurmArchitecture drawio](https://github.com/user-attachments/assets/74fada7b-3d40-43d1-8cec-35c75750787b)

### How SLURM jobs request GPU resources
1. SLURM user submits a job to the SLURM controller with pool and gpu count
1. SLURM ctld uses the prolog script to request GPU resources from the Juice Controller.
   1. On success it stores the encrypted session details on the shared filesystem.
   1. On failure it requeues it to retry later
1. SLURM ctld then schedules the job to a SLURM compute node which loads the encrypted Juice session details from disk and connects to the agent and runs the workload
1. SLURM ctld releases the session with the epilog script

## Implementation
The plugin consists of two parts:


### The Spank Plugin
The plugin exposes Juice options to the slurm user and sets up required settings that are sent to the Prolog scripts for scheduling.

### The Prolog and Epilog Scripts
The Prolog script is responsible for assigning GPUs to the SLURM task. It takes the GPU requests, communicates with the Juice controller and reserves the GPUs for the task. It then exports a data file that the SLURM task uses to run the remote GPUs. If the required GPU resources could not be found, the job will be requeued and tried later.

The Epilog script releases the GPUs when the Job is complete.

## Set up

### Download and Compile
1. Download the Spank plugin and Slurm hooks from this repository.
2. Use `make` to build the plugin with the version of Slurm you intend to use.

```bash
cd spank-plugin
make
```

### Configure the plugin - all nodes
1. Add the `juice.so` file to your slurm nodes and configure it in the slurm `plugstack.conf` with two arguments, the path to your juice binary and a directory for shared state. These paths needs to be accessible on all nodes in the SLURM cluster.

**plugstack.conf**:
```
juice.so /opt/juice/juice /opt/juice/.data
```

### Configure the Prolog and Epilog Scripts - slurmctld

The Prolog script requires a Juice M2M Token to communicate with the Juice Controller. For more details on how to create this token see [M2M Tokens](https://juice-labs.github.io/juice-docs/docs/juice/user-guide/cli-app/m2m-tokens).

Copy the 4 files in the `slurm-hooks` directory to your slurmctld node. Add your M2M Token to the the juice-config.sh file and then update your slurmctld to invoke the Prolog and Epliog scripts.

**slurm.conf**:
```
PrologSlurmctld=/hooks/juice-prolog.sh
EpilogSlurmctld=/hooks/juice-epilog.sh
```

## Usage

To run a SLURM job using a remote GPU the Juice pool must be specified on the SLURM command-line when running `sbatch` or `srun`

Example:
```bash
srun --juice-pool=test hostname -A
```

By default Juice will assign a single GPU, to request additional GPUs add the `--juice-gpu-count` parameter
```bash
srun --juice-pool=test --juice-gpu-count=2 hostname -A
```

**NOTE:**
You cannot use GPUs from multiple agents, there must be a Juice Agent in the organization that has the required number of GPUs or more.

### GPU Slicing

By default Juice will allocate the entire GPU, to allocate a portion of GPU resources specify the `--juice-vram` parameter
```bash
srun --juice-pool=test --juice-vram=1 hostname -A
```
This will allocate 1 GiB of VRAM memory and allow others to take additional slices of VRAM.