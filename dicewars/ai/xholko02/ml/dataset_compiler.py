import numpy as np
import os

DIR = "./dataset/"
NEW_NAME = "./dataset_compiled"
LEN = 634

uninitialized = True

for filename in os.listdir(DIR):
    if filename.endswith(".npy"):
        loc = os.path.join(DIR, filename)
        arr = np.load(loc)

        if uninitialized:
            final_arr = arr
            uninitialized = False
        else:
            final_arr = np.append(final_arr, arr, axis=0)

# save to new file
np.save(NEW_NAME, final_arr)
