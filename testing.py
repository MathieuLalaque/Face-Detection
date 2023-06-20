from latence_functions import *
import numpy as np

array2 = np.array([
    [10,10,0.9],
    [200,200,0.5]
])

array1 = np.array([
    [10,15,0.6],
    [200.5,220,0.4],
    [200,205,0.8]
])

#compare(liste1,liste2,0.7,10,0.5)
#clean(liste2,0.7)




#coords1,score1 = store(array1)
#coords2,score2 = store(array2)

#compare(coords1, score1, coords2, score2)
#print(score1)

#print(coords1[score1>0.5])
#coords1,score1 = clean(coords1,score1,0.5)
#print(coords1,score1)


print(np.linspace(0, 800, 11).astype(np.int32))
print(np.linspace(0, 600, 11).astype(np.int32))

partition = init_partition(10)
print(partition)

latence = create_latence(len(partition))
print(latence)
ping(599,799,partition,latence)
print(latence)
print(partition[599,799])
print(check_latence(100,100,partition,latence))
print(check_latence(599,799,partition,latence))
latence -=1
print(latence)
print(check_latence(100,100,partition,latence))
print(check_latence(599,799,partition,latence))