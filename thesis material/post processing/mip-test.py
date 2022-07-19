import h5py
import matplotlib.pyplot as plt
import numpy as np

def create_mip(fname, data="target", mode="2D", view="axial", norm=False):
    hf = h5py.File(fname, "r")
    vol = hf[data]
    if mode == "2D":
        if len(vol.shape) > 3:
            vol = np.squeeze(vol)
    else:
        vol=vol[:,2,:,:]
    if view == "axial":
        if mode == "2D":
            vol=vol[3:62]
        else:
            vol=vol[1:-1]
    else:
        if mode == "2D":
            vol = vol[20:55]
        else:
            vol = vol[18:53]
    if view == "coronal":
        MIP = np.max(np.abs(vol), 1)
    elif view == "sagittal":
        MIP = np.max(np.abs(vol), 2)
    else:
        MIP = np.max(np.abs(vol), 0)
    if norm:
        MIP = MIP / np.max(MIP)
    hf.close()
    return MIP

# dir = "/path/to/saved/volume"
# name = "t1_3D_5consec_mip"
# mode = "3D"
#
# for view in ["coronal", "sagittal", "axial"]:
#     mip = create_mip(dir, data="reconstruction",mode=mode,view=view)
#     if view != "axial":
#         mip = np.flip(mip,0)
#     plt.imshow(mip, cmap='gray')
#     plt.axis('off')
#     plt.savefig('synthtof_comp/'+name+'_'+view+'.png', bbox_inches='tight', pad_inches=0)

# Recons
dir="/path/to/saved/volume"
data="reconstruction"
name="zerofill"
mode="2D"

for view in ["coronal", "sagittal", "axial"]:
    mips = []
    for i in range(4,0,-1):
        fname = dir+"1_"+str(i)+"_axial_slab"
        mips += [create_mip(fname, data=data, mode=mode, view=view, norm=False)]

    if view == "axial":
        mips = np.stack(mips)
        mip = np.max(mips, 0) / np.max(np.abs(mips))
    else:
        #mips = [m/np.median(np.abs(m)) for m in mips]
        mip = np.concatenate(mips)
    mip = mip / np.max(np.abs(mip))

    plt.imshow(np.abs(mip),cmap='gray')
    plt.axis('off')
    plt.savefig('mips/'+name+'_mip_'+view+'.png', bbox_inches='tight', pad_inches=0)
