import os
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH0='/mnt/ssd0/power_-60dbm/'
number1=13907115668362557567
number2=13835058074324629497
num1=str(hex(number1))
num2=str(hex(number2))
num1=num1[4:18]  #ignore the bits which represent the polarization rather than package
num2=num2[4:18]
num1=int(num1,16)
num2=int(num2,16)
filepath1=PATH0+str(number1)+'.bin' 
filepath2=PATH0+str(number2)+'.bin'
size1=os.path.getsize(filepath1)
print(size1)
size2=os.path.getsize(filepath2)
print(size2)
offset=range(-250,250) #the range of time delay that we want to consider, which should be multiplied by the step
datalength=400000000  #the length of data that we want to use to generate the fringe
step=1               #the step of time delay
file1=open(filepath1,'rb')
file2=open(filepath2,'rb')
if num1>=num2:
    file2.read((num1-num2)*4096)
else:
    file1.read((num2-num1)*4096)   #correct the time delay between the files using the number of the beginning package
data1=np.fromfile(file1,dtype=np.int8,count=datalength+2*step*len(offset),offset=0)   
data2=np.fromfile(file2,dtype=np.int8,count=datalength+2*step*len(offset),offset=0)   # read the data we use, note that we need more length to satisfy maximum time delay, the datatype should also be minded based on ROACH
data1=np.array(data1)
data2=np.array(data2)
data1=data1-data1.mean()  # subtract the offset and long time noise
data2=data2-data2.mean()
file1.close()
file2.close()
intensity=[]
for i in offset:
    print(i)
    if i<0:
        #l=min(size1+i*1*step,size2)       
        #l0=int(l/1000000000)
        store=[] # you can use this list to add up the result from a few segments of data
        for j in range(1):
            data=data1[(-i)*step:datalength-i*step]*data2[0:datalength] # do the correlation by multiplying
            store.append(data.sum()/data.shape[0])
        #data1=[]
        #data2=[]
        #l=l-l0*1000000000
        #for k in range(l):
            #data1.append(int.from_bytes(file1.read(1),'big'))
            #data2.append(int.from_bytes(file2.read(1),'big'))
        #data1=np.array(data1)
        #data2=np.array(data2)
        #data1=data1-data1.mean()
        #data2=data2-data2.mean()
        #data=data1+data2
        #data=data**2
        #store.append(data.sum()/data.shape[0])
        store=np.array(store)
        intensity.append(store.mean())
    else:
        l=min(size1,size2-i*1*step)
        l0=int(l/1000000000)
        store=[]
        for j in range(1):
            data=data1[0:datalength]*data2[(i)*step:datalength+i*step] # the other direction
            store.append(data.sum()/data.shape[0])
        #data1=[]
        #data2=[]
        #l=l-l0*1000000000
        #for k in range(l):
            #data1.append(int.from_bytes(file1.read(1),'big'))
            #data2.append(int.from_bytes(file2.read(1),'big'))
        #data1=np.array(data1)
        #data2=np.array(data2)
        #data1=data1-data1.mean()
        #data2=data2-data2.mean()
        #data=data1+data2
        #data=data**2
        #store.append(data.sum()/data.shape[0])
        store=np.array(store)
        intensity.append(store.mean())
plt.plot(offset,intensity)  
plt.title('the maximum position '+str(np.argmax(intensity)-250))
plt.savefig('noise_test/power_-60dbm.jpg')
plt.close()