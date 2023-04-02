import os
import time

n_runs = 5
job = 'job0.slurm'

for i in range(n_runs):
    time.sleep(10)
    os.system(f'sbatch {job}')

