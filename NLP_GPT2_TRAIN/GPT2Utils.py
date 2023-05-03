
def sample_datasets(data, size=2000):
  
    d = {'train': data[train_split_name].shuffle(seed=42).select(range(size)),
         'validation': data[valid_split_name].shuffle(seed=42).select(range(int(0.33*size)))
        }
    return datasets.dataset_dict.DatasetDict(d)

def read_datasets_hf(d_name_tuple, sample_bool=True):
    #d_name_tuple=('wikitext', 'wikitext-2-raw-v1')
    # tuple len param to avoid error
    if len(d_name_tuple)>1:
        data = load_dataset(d_name_tuple[0], d_name_tuple[1])
    else:
        data = load_dataset(d_name_tuple[0])
    
    if sample_bool:
        data = sample_datasets(data, size=200)
    
    return data

import datasets

def sample_dataset(ds, size=100):
    for i in ds.keys():
        ds[i]=ds[i].shuffle(seed=42).select(range(size))
    return ds

def save_dataset(ds, ds_sample_name):
    ds.save_to_disk(ds_sample_name)
    print("The Dataset is saved")
