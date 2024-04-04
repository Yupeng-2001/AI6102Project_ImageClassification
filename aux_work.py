import os

img_root = f"/mnt/slurm_home/pzzhao/acad_projects/AI6102_proj/dataset"

ct = 0
for root, dirs, files in os.walk(img_root):
    ct += len(files)

print(f"{img_root} count: {ct}")
