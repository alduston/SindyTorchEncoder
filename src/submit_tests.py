import os
import time

n_runs = 3
job = 'job0.slurm'

for i in range(n_runs):
    os.system(f'sbatch {job}')
    time.sleep(10)

