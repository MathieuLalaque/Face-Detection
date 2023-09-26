import os
import numpy as np

os.add_dll_directory("C:\\Users\\lalaq\\Documents\\Travail\\open_cv_shenanigans\\build\\install\\x64\\vc17\\bin")
os.add_dll_directory("C:\\Windows\\System32")
os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin")
import cv2 as cv

import inspect

def func(a,b,c=42) : pass
print(inspect.getargspec(cv.FaceDetectorYN.create))


'''
for root, dirs, files in os.walk("data", topdown=True):
   for name in files:
      print('a')
      print(root.split('\\')[-1])
      print(name[:-4])

   for name in dirs:
      print('b')
      print(root)
      print(name)
      print(os.path.join(root, name))
'''

'''
a = np.zeros((3,10))

a[0,:] = np.arange(1,11)
a[1,:] = np.arange(11,21)
a[2,:] = np.arange(21,31)
indices1 = np.array([0,2])
indices2 = np.array([0,2,7])
print(a)

print(np.where(a[indices1][:,indices2] > 20,0,a[indices1][:,indices2]))


print(a)
print(a[:,0])

'''
'''
im = np.zeros((500,500,3))

im[:,:] = [11,201,176]
im = im/255
print(im)
cv.imshow('televic', im)
cv.waitKey(0)
cv2.destroyAllWindows()
'''