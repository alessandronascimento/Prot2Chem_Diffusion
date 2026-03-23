import pandas as pd 
import selfies as sf 
from datasets import Dataset

def smiles_to_selfies(example):
  try:
    example['selfies'] = sf.encoder(example['smiles'])
  except:
    example['selfies'] = ''
  return example


ds = pd.read_csv('ZINC20_InStock.csv')
ds = Dataset.from_pandas(ds)

ds = ds.map(smiles_to_selfies, num_proc=8)
ds.to_csv('ZINC20_InStock_selfies.csv')

#chembl = pd.read_csv('chembl_36.csv')

#zinc_chembl = pd.concat([ds, chembl])
#print(zinc_chembl)
#zinc_chembl = Dataset.from_pandas(zinc_chembl)
#zinc_chembl = zinc_chembl.train_test_split(test_size=0.2, shuffle=True, seed=13)
#zinc_chembl.to_parquet('zinc20_chembl36.parquet')