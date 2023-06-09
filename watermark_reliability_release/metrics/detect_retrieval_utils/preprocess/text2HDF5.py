import sys
import numpy as np
import h5py
from collections import OrderedDict

f = open(sys.argv[1], 'r')
lines = f.readlines()

lis = []
vocab = OrderedDict()

for i in lines:
    i = i.strip()
    i = i.split('\t')
    if len(i) != int(sys.argv[2]):
        continue
    arr = i[0].split() + i[1].split()
    for j in arr:
        if j not in vocab:
            vocab[j] = len(vocab)

for i in lines:
    i = i.strip()
    i = i.split('\t')
    if len(i) != int(sys.argv[2]):
        continue
    arr = i[0].split()
    s1 = []
    for j in arr:
        s1.append(vocab[j])
    arr1 = np.array(s1, dtype="int32")
    arr = i[1].split()
    s2 = []
    for j in arr:
        s2.append(vocab[j])
    arr2 = np.array(s2, dtype="int32")
    lis.append((arr1, arr2))

arr = np.array(lis)
dt = h5py.vlen_dtype(np.dtype('int32'))

f = h5py.File(sys.argv[1].replace("txt","h5"), 'w')
f.create_dataset("data", data=arr, dtype=dt)

f = open(sys.argv[1].replace("txt","vocab"), 'w')
for i in vocab:
    f.write("{0}\t{1}\n".format(i,vocab[i]))
f.close()
