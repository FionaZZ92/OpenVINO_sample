import os
import pickle

def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'rb') as fid:
        preproc = pickle.load(fid, encoding="latin1")
    return preproc

def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'w') as fid:
        pickle.dump(preproc, fid)
