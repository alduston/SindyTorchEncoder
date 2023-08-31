import os

dir = '../data/misc/dx_plots'
files = os.listdir(dir)
for i,file in enumerate(files):
    os.rename(f'{dir}/{file}', f'{dir}/local_dx_plot{i}.png')
