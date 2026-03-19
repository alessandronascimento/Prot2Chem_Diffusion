import os

def read_sequence(seq_file):
    with open(seq_file, 'r') as f:
        seq = f.readline().strip()
    return seq
 
targets = ['ABL1', 'HDAC8', 'LCK', 'PTN1', 'SRC']

for target in targets:
    target_seq = read_sequence(f'../validation_DUDEZ/{target}/{target}.seq')
    print(f'Target {target} sequence has {len(target_seq)} residues.')
    for i in range(1):
        os.system(f'python3 sampler.py --protein_seq={target_seq} --batch_size=1024 --output_prefix=conditional_generation/DUDEZ/{target}_{i+1} --seed={i} --guidance_scale=10.0')