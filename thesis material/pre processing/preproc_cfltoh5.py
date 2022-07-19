import h5py
import matplotlib.pyplot as plt
import numpy as np

def readcfl(name):
    h = open(name + ".hdr", "r")
    h.readline()  # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split()]
    # remove singleton dimensions from the end
    n = int(np.prod(dims))
    dims_prod = np.cumprod(dims)
    dims = dims[: np.searchsorted(dims_prod, n) + 1]
    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    a = a.reshape(dims, order="F")  # column-major
    return a

ims = []
immax = 0
immded = 0
senses = []
sensemax = 0
sensemed = 0
tars = []
tarmax = 0
tarmed = 0

for i in ["1", "2", "3", "4"]:
    sense = readcfl('/path/to/smap1_'+i)
    kspace = readcfl('/path/to/data1_'+i)

    sense_shifted = np.fft.ifftshift(sense, axes=1)
    sense_shifted = np.transpose(sense_shifted, (2, 3, 0, 1))
    senses += [sense_shifted]
    if np.max(np.abs(sense_shifted)) > sensemax:
        sensemax = np.max(np.abs(sense_shifted))
    #sense_shifted = sense_shifted / np.max(np.abs(sense_shifted))

    imspace = np.fft.ifftn(np.fft.fftshift(kspace, axes=(0, 1, 2)), axes=(0, 1, 2), norm="forward")
    imspace_shifted = np.fft.ifftshift(imspace, axes=0)
    imspace_shifted = np.transpose(imspace_shifted, (2, 3, 0, 1))
    ims += [imspace_shifted]
    if np.max(np.abs(imspace_shifted)) > immax:
        immax = np.max(np.abs(imspace_shifted))
    #imspace_shifted = imspace_shifted / np.max(np.abs(imspace_shifted))

    target = np.sum(imspace_shifted*sense_shifted.conj(), 1)
    tars += [target]
    if np.max(np.abs(target)) > tarmax:
        tarmax = np.max(np.abs(target))
    #target = target / np.max(np.abs(target))


for i in range(4):
    fname="/path/to/save/1_"+str(i+1)+"_axial_slab"

    # SLABNORM median
    # imspace_shifted = ims[i]/np.median(np.abs(ims[i]))
    # sense_shifted = senses[i]/np.median(np.abs(senses[i]))
    # target = tars[i]/np.median(np.abs(tars[i]))

    # VOLNORM
    # imspace_shifted = ims[i]/immax
    # sense_shifted = senses[i]/sensemax
    # target = tars[i]/tarmax

    imspace_shifted = ims[i]
    sense_shifted = senses[i]
    target = tars[i]

    # SLICENORM
    # for j in range(target.shape[0]):
    #     imspace_shifted[j] = imspace_shifted[j]/np.max(np.abs(imspace_shifted[j]))
    #     sense_shifted[j] = sense_shifted[j]/np.max(np.abs(sense_shifted[j]))
    #     target[j] = target[j]/np.max(np.abs(target[j]))

    hf = h5py.File(fname, "w")
    hf.create_dataset("kspace", data=np.fft.fftn(imspace_shifted, axes=(-2, -1)).astype(np.complex64))
    hf.create_dataset("sensitivity_map", data=sense_shifted.astype(np.complex64))
    hf.create_dataset("target", data=np.abs(target).astype(np.float32))
    hf.close()

    plt.imshow(np.abs(target[10,:,:]), cmap='gray')
    plt.show()
