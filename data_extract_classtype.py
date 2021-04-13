import numpy as np
from datetime import datetime
print('{}-Start!'.format(datetime.now()))
data = np.loadtxt('E:\\RK\\Toronto_3D\\val\\L002 - Cloud.txt',dtype= str)
labels = data[:,6]
labels = labels.reshape((-1,1))
np.savetxt('E:\\RK\\Toronto_3D\\val\\L002 - Cloud.labels',labels ,fmt='%s')
print('{}-Done!'.format(datetime.now()))
