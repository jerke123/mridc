import h5py
import matplotlib.pyplot as plt
import numpy as np

fname = "/path/to/file"
data = "target"
slice = 20

hf = h5py.File(fname, "r")
print(list(hf.keys()))
#vol = np.sum(np.fft.ifftn(hf[data][slice],axes=(1,2)) * hf["sensitivity_map"][slice].conj(),0)
vol = hf[data][slice]

# fig=plt.imshow(np.abs(vol),cmap='gray')
# plt.axis('off')
# plt.savefig('synthplots/magnitude_synth.png', bbox_inches='tight',pad_inches = 0)
#
# fig=plt.imshow(np.angle(vol),cmap='gray')
# plt.axis('off')
# plt.savefig('synthplots/phase_synth.png', bbox_inches='tight',pad_inches = 0)
#
# fig=plt.imshow(vol.real,cmap='gray')
# plt.axis('off')
# plt.savefig('synthplots/real_synth.png', bbox_inches='tight',pad_inches = 0)
#
# fig=plt.imshow(vol.imag,cmap='gray')
# plt.axis('off')
# plt.savefig('synthplots/imag_synth.png', bbox_inches='tight',pad_inches = 0)

fig, axs = plt.subplots(2, 2)
fig.suptitle('recon')

axs[0, 0].imshow(np.abs(vol),cmap='gray')
axs[0, 0].set_title("magnitude")
axs[0, 0].axis('off')

axs[0, 1].imshow(np.angle(vol),cmap='gray')
axs[0, 1].set_title("phase")
axs[0, 1].axis('off')

axs[1, 0].imshow(vol.real,cmap='gray')
axs[1, 0].set_title("real")
axs[1, 0].axis('off')

axs[1, 1].imshow(vol.imag,cmap='gray')
axs[1, 1].set_title("imag")
axs[1, 1].axis('off')

plt.show(bbox_inches='tight', pad_inches = 0)
