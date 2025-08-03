# import library
from dataset import DS
from iv import iv_main
import glob
import os
datasets_dir = "./datasets/"
# we can easily load all of the `datasets/*.csv` files using glob and pass their
# file path to the `DS` type which expects `DS(path1, path2, name`, since we are doing the normal
# 70-30 ratio, we only need to pass for `path1` and let `path2 be `None`, we also let the file's base name be
# the name to pass to `DS`
datasets = [DS(d, None, os.path.splitext(os.path.basename(d))[0]) for d in glob.glob(f"{datasets_dir}/*.csv")]
# Define `params.json` as the path which includes the hyperparameters:
# - lr
# - weight_decay
# - batch_size
# - epochs
# - dropout_rate
params_path = 'params.json'
for dataset in datasets:
    # Creates a tuple of (OptionDataset, OptionDataset) which is what is used in the dataloader
    # The first is the training dataset, and the second is the test
    ds_train, ds_test = dataset.datasets()
    # The ao_ann_main function is the main entry, and does the training, testing, and saves the model
    iv_main(ds_train, ds_test, dataset.name, params_path)
