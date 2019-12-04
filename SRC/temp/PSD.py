#Feature extraction :

import numpy as np
import scipy.io as spi 
#from biosppy.signals.eeg import get_power_features
from scipy.signal import welch
from time import time
import pyeeg
import pywt
import nitime

import math
from scipy.signal import correlate
from numpy.fft import fft
from scipy import stats as st

#The files where the data will be stored
outMatrix=['Right.mat','Left.mat','Up.mat','Down.mat']

#to check if all 3 speeds are done
flagr=0
flagl=0
flagu=0
flagd=0

ind1=[0, 1]
ind2=[2, 3]
features=[]

#All the keys in the dictionary
direc=['rs1','rs2','rs3','ls1','ls2','ls3','us1','us2','us3','ds1','ds2','ds3']

for h in range(len(direc)):#participants, for us its 4 directions
    #print(h)
    #Initialization of array space to store the features.
    m=final_data[direc[h]]
    featuresPSD=np.ndarray(shape=(27,12))

    featuresPSD1=np.ndarray(shape=(27,12))

    datData=m

    for i in range(27):#trials


        localPSD=[]
        localPSD1=[]

        #if its not speed1 i.e only 256 values
        if(direc[h][2]!='1'):
       
            for j in range(4):#channels
                arr=m[i][j][0:256]
                freq,PSD=welch(arr)#calculates PSD coeffiecnts n freqs
                PSD=PSD.tolist()
                maxFreq=freq[PSD.index(max(PSD))]
                localPSD=localPSD+[np.mean(PSD),np.var(PSD),maxFreq]#appending features of all channels
                

            #
            
            featuresPSD[i,:]=localPSD#adding features of each trial
            
        else:#if it has more than 256(and 512) values
         
            for j in range(4):#channels
                arr=m[i][j][0:256]#0 to 256 values
                arr1=m[i][j][256:512]#256 to 512 values
                freq1,PSD1=welch(arr)
                PSD1=PSD1.tolist()
                maxFreq1=freq1[PSD1.index(max(PSD1))]
                localPSD1=localPSD1+[np.mean(PSD1),np.var(PSD1),maxFreq1]
                arr=m[i][j][0:256]
                freq,PSD=welch(arr)
                PSD=PSD.tolist()
                maxFreq=freq[PSD.index(max(PSD))]
                localPSD=localPSD+[np.mean(PSD),np.var(PSD),maxFreq]

          
            #adding features of each trial
            featuresPSD[i,:]=localPSD
            featuresPSD1[i,:]=localPSD1
     
    if(direc[h][0]=='r'):
        if(flagr==0):#first time reading a particular direction
            s={'PSD':[]}
            s['PSD']=featuresPSD
            s['PSD']=np.vstack((s['PSD'],featuresPSD1))

            flagr+=1
        else:
            s['PSD']=np.vstack((s['PSD'],featuresPSD))       
            flagr+=1
            if(flagr==3):#last file in the particular direction
                spi.savemat(outMatrix[0],s)
                flagr=0


    if(direc[h][0]=='l'):
        if(flagl==0):#first time reading a particular direction
            s={'PSD':[]}
            s['PSD']=featuresPSD
            s['PSD']=np.vstack((s['PSD'],featuresPSD1))
            flagl+=1
        else:
            #s['hjorth'].numpy.append(featuresHjorth)
            s['PSD']=np.vstack((s['PSD'],featuresPSD))
            flagl+=1
            if(flagl==3):#last file in the particular direction
                spi.savemat(outMatrix[1],s)
                flagl=0




    if(direc[h][0]=='u'):
        if(flagu==0):#first time reading a particular direction
            s={'PSD':[]}
            s['PSD']=featuresPSD
            s['PSD']=np.vstack((s['PSD'],featuresPSD1))
            flagu+=1
        else:
            s['PSD']=np.vstack((s['PSD'],featuresPSD))
            flagu+=1
            if(flagu==3):#last file in the particular direction
                spi.savemat(outMatrix[2],s)
                flagu=0



    if(direc[h][0]=='d'):
        if(flagd==0):#first time reading a particular direction
            s={'PSD':[]}
            s['PSD']=featuresPSD
            s['PSD']=np.vstack((s['PSD'],featuresPSD1))
            flagd+=1
        else:
            s['PSD']=np.vstack((s['PSD'],featuresPSD))
            flagd+=1
            if(flagd==3):#last file in the particular direction
                spi.savemat(outMatrix[3],s)
                flagd=0

