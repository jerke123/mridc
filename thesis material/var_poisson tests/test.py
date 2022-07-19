import numpy as np
import time

# import transforms as T
# from subsample import Gaussian2DMaskFunc
from poisson_test import poisson_disc_pattern
from sigpy.mri.samp import poisson
from matplotlib import pyplot as plt

# mask_func = Gaussian2DMaskFunc(center_fractions=[0.90], accelerations=[16])
# mask = mask_func(shape=[16,640,640,8])[2]
# print(np.sum(mask))

start = time.process_time_ns()
#mask= poisson_disc_pattern([300,300], radius=10, r=70, center=True)
mask= poisson([300,300], 7.2)
end = time.process_time_ns()
s = (end-start)/10**9
print("Time elapsed : " + str(s) + " sec")
print("Subsample factor: " + str(mask.size/mask.sum()))
plt.imshow(np.abs(mask), cmap='gray')
plt.show()
