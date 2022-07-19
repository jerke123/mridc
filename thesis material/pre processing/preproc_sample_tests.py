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

def createh5(sense_shifted,kspace,npnorm="backward",kshift=True,maxdiv=True):

    if kshift:
        imspace = np.fft.ifftn(np.fft.fftshift(kspace, axes=(0, 1, 2)), axes=(0, 1, 2), norm=npnorm)[:,:,20:26,:]
    else:
        imspace = np.fft.ifftn(kspace, axes=(0, 1, 2), norm=npnorm)[:,:,20:26,:]

    imspace_shifted = np.fft.ifftshift(imspace, axes=0)
    imspace_shifted = np.transpose(imspace_shifted, (2, 3, 0, 1))

    if maxdiv:
        imspace_shifted = imspace_shifted / np.max(np.abs(imspace_shifted))
        sense_shifted = sense_shifted / np.max(np.abs(sense_shifted))
    target = np.abs(np.sum(imspace_shifted * sense_shifted.conj(), 1))
    if maxdiv:
        target = target / np.max(np.abs(target))


    fname="???"\
          +npnorm+"_kshift_"+str(kshift)+"_maxdiv_"+str(maxdiv)
    hf = h5py.File(fname, "w")
    hf.create_dataset("kspace", data=np.fft.fftn(imspace_shifted, axes=(-2, -1)).astype(np.complex64))
    hf.create_dataset("sensitivity_map", data=sense_shifted.astype(np.complex64))
    hf.create_dataset("target", data=target.astype(np.float32))
    hf.close()

sense = readcfl("???")[:,:,20:26,:]
sense_shifted = np.fft.ifftshift(sense, axes=1)
sense_shifted = np.transpose(sense_shifted, (2, 3, 0, 1))

kspace = readcfl("???")

npnorms=["backward","ortho","forward"]
kshifts=[True,False]
maxdivs=[True,False]

for n in npnorms:
    for k in kshifts:
        for m in maxdivs:
            createh5(sense_shifted,kspace,npnorm=n,kshift=k,maxdiv=m)


# #imspace = np.fft.ifftn(np.fft.fftshift(kspace, axes=(0, 1, 2)), axes=(0, 1, 2), norm="ortho")[:,:,20:25,:]
# imspace = np.fft.ifftn(kspace, axes=(0, 1, 2), norm="forward")[:,:,20:25,:]
#
#
# imspace_shifted = np.fft.ifftshift(imspace, axes=0)
# imspace_shifted = np.transpose(imspace_shifted, (2, 3, 0, 1))
# imspace_shifted = imspace_shifted / np.max(np.abs(imspace_shifted))
#
# sense_shifted = sense_shifted / np.max(np.abs(sense_shifted))
#
# target = np.abs(np.sum(imspace_shifted*sense_shifted.conj(), 1))
# target = target / np.max(np.abs(target))
#
# hf = h5py.File("???", "w")
# hf.create_dataset("kspace", data=np.fft.fftn(imspace_shifted, axes=(-2, -1)).astype(np.complex64))
# hf.create_dataset("sensitivity_map", data=sense_shifted.astype(np.complex64))
# hf.create_dataset("target", data=target.astype(np.float32))
# hf.close()
#
# plt.imshow(np.abs(target[3,:,:]), cmap='gray')
# plt.show()
