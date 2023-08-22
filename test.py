import os
import numpy as np
import cv2 as cv
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

for i in a:
    print(i)
'''

cv.release()



