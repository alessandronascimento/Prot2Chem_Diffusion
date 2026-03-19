from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('alessandronascimento/clustered_jglaser_binding_affinity', split='train')

with open('jglaser.fasta', 'w') as f:
    for i in tqdm(range(len(dataset))):
        prot_seq = dataset[i]['seq']
        f.write(f'>{i}\n{prot_seq}\n')

print('Dataset written as jglaser.fasta.')


