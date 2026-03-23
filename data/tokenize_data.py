from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from datasets import DatasetDict
from huggingface_hub import login

# MolGen tokenizer
tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large")

def is_valid(example):
    try:
        tokenizer(example["selfies"])
        return True
    except Exception:
        return False

def correct_ZINC_id(example):
    example['id'] = f"ZINC_{example['id']}"
    return example

print('Loading Zinc dataset...')
zinc = load_dataset('csv', data_files='ZINC20_InStock_selfies.csv')
zinc = zinc.filter(lambda x: x["selfies"] != '')

print('Loading Chembl dataset...')
chembl = load_dataset('csv', data_files='chembl_36.csv')
chembl = chembl.filter(lambda x: x["selfies"] != '')

print('Mapping ZINC to correct Ids...')
zinc = zinc.map(correct_ZINC_id, num_proc=8)

print('Concatenating datasets...')
ds = concatenate_datasets([zinc['train'],chembl['train']])

print('Tokenizing dataset...')
ds = ds.filter(is_valid, num_proc=8)
ds = ds.map( lambda x: tokenizer(x["selfies"]), batched=True, num_proc=8)


print(f'Final dataset length: {len(ds)}')

# Make splits

ds_split = ds.train_test_split(test_size=0.2, seed=13, shuffle=True)

train_ds = ds_split["train"]
temp_ds = ds_split["test"]

temp_split = temp_ds.train_test_split(test_size=0.5, seed=13)

val_ds = temp_split["train"]
test_ds = temp_split["test"]

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds,
})

dataset.save_to_disk('tokenized_dataset')
