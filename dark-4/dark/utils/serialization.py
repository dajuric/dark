import os
import pickle

def load(filename):
    obj = pickle.load(open(filename, "rb"))
    return obj

def save(obj, filename):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    pickle.dump(obj, open(filename, "wb"))