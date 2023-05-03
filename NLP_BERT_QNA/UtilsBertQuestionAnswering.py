import datasets

def sample_dataset(ds, size=100):
    for i in ds.keys():
        ds[i]=ds[i].shuffle(seed=42).select(range(size))
    return ds

def save_dataset(ds_sample_name):
    ds.save_to_disk("squad-sample")
    print("The Dataset is saved")
    
