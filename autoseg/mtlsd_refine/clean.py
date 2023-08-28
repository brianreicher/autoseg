import numpy as np
import zarr
from skimage.morphology import remove_small_objects
from skimage.measure import label

for thresh in [0.6,0.7]:

    seg_ds = f"seg_{thresh}"

    #f = zarr.open('validation_prediction.zarr', 'a')
    f = zarr.open('test_prediction.zarr', 'a')

    print("loading")
    seg = f[seg_ds][:]

    print("getting uniques")
    unique = np.unique(seg, return_counts=True)
    print(unique)

    remove = []

    for i, j in zip(unique[0], unique[1]):
        if j > 3500000:
            remove.append(i)

    print("removing large uniques")
    seg[np.isin(seg, remove)] = 0

    f[f'seg_{thresh}_filtered'] = seg
    f[f'seg_{thresh}_filtered'].attrs['offset'] = f[seg_ds].attrs['offset']
    f[f'seg_{thresh}_filtered'].attrs['resolution'] = f[seg_ds].attrs['resolution']
