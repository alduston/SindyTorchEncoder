import os
import time

n_runs = 20
job = 'job1.slurm'

for i in range(n_runs):
    os.system(f'sbatch {job}')
    time.sleep(10)

