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

tofs = ['files', 'in', 'here']
views = ['axial','sagittal','coronal']

for tof in tofs:
    dir = '/.../'+tof
    sense = readcfl(dir+'/4RIM/tof/sensemaps')
    kspace = readcfl(dir+'/4RIM/tof/data')

    imspace = np.fft.ifftn(np.fft.fftshift(kspace, axes=(0, 1, 2)), axes=(0, 1, 2), norm="forward")
    imspace_shift = np.fft.ifftshift(imspace, axes=(0, 1, 2))

    for view in views:
        fname = "/savepath/"+tof+"_"+view

        # sense / imspace : (H,W,Z,C)
        if view == "axial":
            imspace_shifted = np.transpose(imspace_shift, (2, 3, 0, 1))
            sense_shifted = np.transpose(sense, (2, 3, 0, 1))
        if view == "sagittal":
            imspace_shifted = np.transpose(imspace_shift, (1, 3, 2, 0))
            imspace_shifted = np.flip(imspace_shifted, axis=2)
            imspace_shifted = imspace_shifted[60:-60,:,:,:]
            sense_shifted = np.transpose(sense, (1, 3, 2, 0))
            sense_shifted = np.flip(sense_shifted, axis=2)
            sense_shifted = sense_shifted[60:-60,:,:,:]
        if view == "coronal":
            imspace_shifted = np.transpose(imspace_shift, (0, 3, 2, 1))
            imspace_shifted = np.flip(imspace_shifted, axis=2)
            imspace_shifted = imspace_shifted[60:-60, :, :, :]
            sense_shifted = np.transpose(sense, (0, 3, 2, 1))
            sense_shifted = np.flip(sense_shifted, axis=2)
            sense_shifted = sense_shifted[60:-60, :, :, :]

        sense_shifted = sense_shifted / np.max(np.abs(sense_shifted))
        imspace_shifted = imspace_shifted / np.max(np.abs(imspace_shifted))

        target = imspace_shifted * sense_shifted.conj()

        phase = np.tile(np.linspace(-np.pi,np.pi,int(sense_shifted.shape[-2]/2))[:,None],(sense_shifted.shape[-1],2,sense_shifted.shape[-3])).T
        gauss = np.random.normal(0, 0.01, phase.shape) + 1j * np.random.normal(0, 0.01, phase.shape)

        target = np.sum(imspace_shifted*sense_shifted.conj(),1)
        target = target / np.max(np.abs(target))

        imspace_shifted = imspace_shifted * np.exp(1j*phase) + gauss
        imspace_shifted = imspace_shifted / np.max(np.abs(imspace_shifted))

        kspace_shifted = np.fft.fftn(imspace_shifted, axes=(2,3))

        hf = h5py.File(fname, "w")
        hf.create_dataset("kspace", data=kspace_shifted.astype(np.complex64))
        hf.create_dataset("sensitivity_map", data=sense_shifted.astype(np.complex64))
        hf.create_dataset("target", data=np.abs(target).astype(np.float32))
        hf.close()

        plt.imshow(np.abs(target[80]), cmap='gray')
        plt.show()
