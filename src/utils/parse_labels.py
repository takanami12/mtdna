import csv
import re
import os
import glob

# Low level check for what columns we have
with open('/home/thovd/mtdna/metadata_rerun.tsv', 'r') as f:
    lines = f.readlines()

header = lines[0].strip().split('\t')
print("Columns:", len(header), header)

# We want to trace Run to batch folder?
# Actually, we can just list all JSON files first.
batch_dirs = glob.glob('/home/thovd/mtdna/pipeline_results/*')
json_files = glob.glob('/home/thovd/mtdna/pipeline_results/*/*/*.json')
print(f"Total JSON files: {len(json_files)}")

# Let's map sample to its json files
sample_to_runs = {}
for jf in json_files:
    parts = jf.split('/')
    batch = parts[-3]
    sample = parts[-2]
    filename = parts[-1]
    
    if sample not in sample_to_runs:
        sample_to_runs[sample] = []
    sample_to_runs[sample].append(jf)

# print a few
print("A few json files:", json_files[:3])
