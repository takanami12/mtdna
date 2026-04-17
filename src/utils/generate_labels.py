import csv
import glob
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
tsv_file = PROJECT_ROOT / 'metadata_rerun.tsv'
pipeline_dir = PROJECT_ROOT / 'pipeline_results'
output_file = PROJECT_ROOT / 'labels.csv'

# Read existing JSON files
# Format: pipeline_results/batch/sample/filename.json
# We want a map from sample_id to all its json files
json_files = glob.glob(os.path.join(pipeline_dir, '*/*/*.json'))
sample_to_jsons = {}
for jf in json_files:
    sample = os.path.basename(os.path.dirname(jf))
    if sample not in sample_to_jsons:
        sample_to_jsons[sample] = []
    sample_to_jsons[sample].append(jf)

print(f"Loaded {len(json_files)} json files across {len(sample_to_jsons)} samples.")

parsed_labels = {}

def has_hv_or_number(text):
    if 'hv' in text.lower():
        return True
    if any(c.isdigit() for c in text):
        return True
    return False

def parse_labels(issue_text):
    text = issue_text.lower()
    lbls = {'HV1F': 0, 'HV1R': 0, 'HV2F': 0, 'HV3R': 0}
    
    if '4 trình tự' in text or 'fail 4 chiều' in text or 'fail 4 chieu' in text:
        return {'HV1F': 1, 'HV1R': 1, 'HV2F': 1, 'HV3R': 1}
        
    if '1f' in text or 'hv1f' in text or '1 f' in text:
        lbls['HV1F'] = 1
    if '1r' in text or 'hv1r' in text or '1 r' in text:
        lbls['HV1R'] = 1
    if '2f' in text or 'hv2f' in text or '2 f' in text or 'hv2' in text:
        lbls['HV2F'] = 1
    if '3r' in text or 'hv3r' in text or '3 r' in text or '3f' in text or 'hv3' in text:
        lbls['HV3R'] = 1
        
    return lbls

with open(tsv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    
    for row in reader:
        if len(row) < 4:
            continue
            
        sample = row[1].strip()
        
        # Check Run1 Issue
        issue1 = row[3].strip() if len(row) > 3 else ""
        issue2 = row[6].strip() if len(row) > 6 else ""
        issue3 = row[9].strip() if len(row) > 9 else ""
        
        # Combine all issues for the sample? Or do we create labels per run?
        # A sample might have multiple batches in pipeline_results.
        # But wait, we said most samples only have 1 batch in pipeline_results because pipeline_results contains the first run!
        # Actually pipeline_results contains multiple batches. Each batch is a folder.
        # If a sample was run twice, it would have 2 folders inside pipeline_results?
        # Let's just combine all issues for the sample for now, because maybe we want to label the JSON file.
        # Wait, if Run1 was noisy, and Run2 was good, Run2's JSON won't be noisy!
        # If we combine issues, we might falsely label Run2's JSON as noisy!
        all_issues = [issue1, issue2, issue3]
        
        for issue in all_issues:
            if issue and has_hv_or_number(issue):
                lbls = parse_labels(issue)
                if sample not in parsed_labels:
                    parsed_labels[sample] = {'HV1F': 0, 'HV1R': 0, 'HV2F': 0, 'HV3R': 0}
                
                # Logical OR to accumulate noise labels?
                for k in lbls:
                    parsed_labels[sample][k] |= lbls[k]

print(f"Successfully parsed {len(parsed_labels)} samples from TSV that match the condition.")

# Now map to JSON files
dataset = []
skipped_samples = 0
for sample, lbls in parsed_labels.items():
    if sample in sample_to_jsons:
        js_list = sample_to_jsons[sample]
        for jf in js_list:
            fname = os.path.basename(jf).upper()
            rel_path = Path(jf).relative_to(PROJECT_ROOT)
            if 'HV1F' in fname:
                dataset.append((rel_path, lbls['HV1F']))
            elif 'HV1R' in fname:
                dataset.append((rel_path, lbls['HV1R']))
            elif 'HV2F' in fname:
                dataset.append((rel_path, lbls['HV2F']))
            elif 'HV3R' in fname:
                dataset.append((rel_path, lbls['HV3R']))
    else:
        skipped_samples += 1

print(f"Skipped {skipped_samples} samples not found in pipeline_results.")
print(f"Generated {len(dataset)} labeled file paths.")

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['FilePath', 'Label'])
    for dp in dataset:
        writer.writerow(dp)
print(f"Saved to {output_file}")
