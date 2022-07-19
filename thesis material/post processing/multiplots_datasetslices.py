import h5py
import matplotlib.pyplot as plt
import numpy as np

def saveslice(name, fname, type="magnitude", data="reconstruction", slice=75):
    hf = h5py.File(fname, "r")
    im = hf[data][slice]
    if len(im.shape) > 2:
        im = im[0]
    if type == "magnitude":
        im = np.abs(im)
        im = im/np.max(im)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.savefig('2dt1comparison/'+name+'.png', bbox_inches='tight', pad_inches=0)

saveslice("ground_truth", "/path/to/file", data="target")
