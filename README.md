# Project 2

Name: Annabelle Huang

CNET ID: ahuang02

To build and run the ray tracing programs, use the command `make ray_tracing`.

### Serial Ray Tracing

- Let `partition = caslake` and `cpus-per-task = 1`. Use the command `srun ./ray_tracing_serial <NRAYS> <NGRID> 1` in the batchfile.

### Multicore Ray Tracing

- Let `partition = caslake` and `cpus-per-task = 16`. Use the command `srun ./ray_tracing_openmp <NRAYS> <NGRID> 16` in the batchfile.

### GPU Ray Tracing

- Let `partition = gpu` and `gres = gpu:1`. Use the command `srun ./ray_tracing_gpu_v100 <NRAYS> <NGRID> <NBLOCKS> <NTPB>` and `constraint = v100` in the batchfile for the V100 processor. Use `ray_tracing_gpu_a100` and `constraint = a100` for the corresponding A100 processor. Use `ray_tracing_gpu_rtx6000` and `constraint = rtx6000` for the corresponding RTX 6000 processor.

### Multi-GPU Ray Tracing

- Let `partition = gpu` and `gres = gpu:2` and `ntasks-per-node = 2`, and `gpus-per-task = 1`. Use the command `mpirun ./ray_tracing_multi <NRAYS> <NGRID> <NBLOCKS> <NTPB>` and `constraint = v100` in the batchfile for the V100 processors. Use `constraint = a100` and `constraint = rtx6000` for the corresponding A100 and RTX 6000 processors.
