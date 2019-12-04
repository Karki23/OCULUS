
# coding: utf-8

# In[3]:

import glob
import os
import numpy as np
get_ipython().magic('matplotlib inline')
from numpy import sqrt
import matplotlib.pyplot as plt
import scipy as sp
import math
import scipy
import scipy.io as io
import scipy.signal as sig
import signal
import csv

file_list = glob.glob("*.txt") #takes a regex to take all text files

#seperate the data into speed and direction. So 12 keys and final_data is a dictionary
final_data={'rs1':[],'rs2':[],'rs3':[],'ls1':[],'ls2':[],'ls3':[],'us1':[],'us2':[],'us3':[],'ds1':[],'ds2':[],'ds3':[]}


#intiliaze the values of 12 keys. Each value is a numpy array
#The minimum number of samples from all possible trials are 771 for speed1, 397 for speed2, and 259 for speed3
d0arr=np.zeros((27,4,771))
d1arr=np.zeros((27,4,397))
d2arr=np.zeros((27,4,259))

u0arr=np.zeros((27,4,771))
u1arr=np.zeros((27,4,397))
u2arr=np.zeros((27,4,259))

l0arr=np.zeros((27,4,771))
l1arr=np.zeros((27,4,397))
l2arr=np.zeros((27,4,259))

r0arr=np.zeros((27,4,771))
r1arr=np.zeros((27,4,397))
r2arr=np.zeros((27,4,259))


arr=[]
trial=0
flagdown=0
for file_name in file_list:
    print(file_name)
    i=0
    j=0
    array=[]
    c1=[]
    c2=[]
    c3=[]
    c4=[]
    file=open(file_name,"r")
    for line in file:
        if i>=1280: #First 5 seconds needs to removed; noise in OpenBCI GUI. 5 seconds x 256 = 1280 samples
            array=line.split(",")
            j=j+1
            c1.append(float(array[1]))
            c2.append(float(array[2]))
            c3.append(float(array[3]))
            c4.append(float(array[4]))
        i=i+1
        # Detrending. Commented that as it was giving poor results.
#     c1=scipy.signal.detrend(c1, axis=-1, type='linear', bp= [j for j in range(0,len(c1),259)])
#     c2=scipy.signal.detrend(c2, axis=-1, type='linear', bp= [j for j in range(0,len(c1),259)])
#     c3=scipy.signal.detrend(c3, axis=-1, type='linear', bp= [j for j in range(0,len(c1),259)])
#     c4=scipy.signal.detrend(c4, axis=-1, type='linear', bp= [j for j in range(0,len(c1),259)])

        
    speed=int(file_name[9])-1
        
    direction=file_name[6]
    
    trial_char=file_name
    trial=3*(int(trial_char[11])-1)
    speed=int(file_name[9])-1
    loc=int(file_name[7])-1 #loc represents which poistion; up or middle or down
    
    if(loc==0):
        if(len(file_name)>17):
            if(file_name[14]=='2'):
                addtrial=1
            elif(file_name[14]=='3'):
                addtrial=2
        else:
            addtrial=0
        trial=trial+addtrial
              
    if(loc==1):
        if(len(file_name)>17):
            if(file_name[14]=='2'):
                addtrial=1
            elif(file_name[14]=='3'):
                addtrial=2
        else:
            addtrial=0
        trial=(trial+addtrial)+9
        
    if(loc==2):
        if(len(file_name)>17):
            if(file_name[14]=='2'):
                addtrial=1
            elif(file_name[14]=='3'):
                addtrial=2
        else:
            addtrial=0
        trial=(trial+addtrial)+18
    
    
    if direction == 'r':
        dir=1
        if speed==0:
            c1=c1[0:771]
            c2=c2[0:771]
            c3=c3[0:771]
            c4=c4[0:771]
            r0arr[trial][0]=c1
            r0arr[trial][1]=c2
            r0arr[trial][2]=c3
            r0arr[trial][3]=c4
        elif speed==1:
            c1=c1[0:397]
            c2=c2[0:397]
            c3=c3[0:397]
            c4=c4[0:397]
            r1arr[trial][0]=c1
            r1arr[trial][1]=c2
            r1arr[trial][2]=c3
            r1arr[trial][3]=c4
        elif speed==2:
            c1=c1[0:259]
            c2=c2[0:259]
            c3=c3[0:259]
            c4=c4[0:259]
            r2arr[trial][0]=c1
            r2arr[trial][1]=c2
            r2arr[trial][2]=c3
            r2arr[trial][3]=c4
            
    elif direction =='d':
        dir=2
        if speed==0:
            c1=c1[0:771]
            c2=c2[0:771]
            c3=c3[0:771]
            c4=c4[0:771]
            d0arr[trial][0]=c1
            d0arr[trial][1]=c2
            d0arr[trial][2]=c3
            d0arr[trial][3]=c4

        elif speed==1:
            c1=c1[0:397]
            c2=c2[0:397]
            c3=c3[0:397]
            c4=c4[0:397]
            d1arr[trial][0]=c1
            d1arr[trial][1]=c2
            d1arr[trial][2]=c3
            d1arr[trial][3]=c4
        elif speed==2:
            c1=c1[0:259]
            c2=c2[0:259]
            c3=c3[0:259]
            c4=c4[0:259]
            d2arr[trial][0]=c1
            d2arr[trial][1]=c2
            d2arr[trial][2]=c3
            d2arr[trial][3]=c4
            
    if direction == 'l':
        dir=3
        if speed==0:
            c1=c1[0:771]
            c2=c2[0:771]
            c3=c3[0:771]
            c4=c4[0:771]
            l0arr[trial][0]=c1
            l0arr[trial][1]=c2
            l0arr[trial][2]=c3
            l0arr[trial][3]=c4
        elif speed==1:
            c1=c1[0:397]
            c2=c2[0:397]
            c3=c3[0:397]
            c4=c4[0:397]
            l1arr[trial][0]=c1
            l1arr[trial][1]=c2
            l1arr[trial][2]=c3
            l1arr[trial][3]=c4
        elif speed==2:
            c1=c1[0:259]
            c2=c2[0:259]
            c3=c3[0:259]
            c4=c4[0:259]
            l2arr[trial][0]=c1
            l2arr[trial][1]=c2
            l2arr[trial][2]=c3
            l2arr[trial][3]=c4
            
    elif direction =='u':
        dir=4
        if speed==0:
            c1=c1[0:771]
            c2=c2[0:771]
            c3=c3[0:771]
            c4=c4[0:771]
            u0arr[trial][0]=c1
            u0arr[trial][1]=c2
            u0arr[trial][2]=c3
            u0arr[trial][3]=c4
        elif speed==1:
            c1=c1[0:397]
            c2=c2[0:397]
            c3=c3[0:397]
            c4=c4[0:397]
            u1arr[trial][0]=c1
            u1arr[trial][1]=c2
            u1arr[trial][2]=c3
            u1arr[trial][3]=c4
        elif speed==2:
            c1=c1[0:259]
            c2=c2[0:259]
            c3=c3[0:259]
            c4=c4[0:259]
            u2arr[trial][0]=c1
            u2arr[trial][1]=c2
            u2arr[trial][2]=c3
            u2arr[trial][3]=c4


            
final_data['rs1']=r0arr
final_data['rs2']=r1arr
final_data['rs3']=r2arr

final_data['ls1']=l0arr
final_data['ls2']=l1arr
final_data['ls3']=l2arr

final_data['us1']=u0arr
final_data['us2']=u1arr
final_data['us3']=u2arr

final_data['ds1']=d0arr
final_data['ds2']=d1arr
final_data['ds3']=d2arr





# In[2]:

sp.io.savemat('final.mat',final_data) # Save the dictionary to a mat file.


# In[ ]:



