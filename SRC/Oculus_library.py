#Preprocess
import glob
import os
import numpy as np
from numpy import sqrt
import scipy as sp



#Bandpass
import scipy.io as io
import scipy.signal as sig
import signal
from scipy.signal import butter,lfilter



#PSD
from scipy.signal import welch
import pyeeg


#Classifier
import _pickle
import scipy
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import statistics 


fs=256
nyq = 0.5 * fs
lowcut=0.5
highcut=30
low = lowcut / nyq
high = highcut / nyq

#seperate the data into speed and direction. So 12 keys and final_data is a dictionary
final_data={'rs1':[],'rs2':[],'rs3':[],'ls1':[],'ls2':[],'ls3':[],'us1':[],'us2':[],'us3':[],'ds1':[],'ds2':[],'ds3':[]}

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y
















def preprocess():
    file_list = glob.glob("Input_files/Karki_data/*.txt") #takes a regex to take all text files
    


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

        
        file_name=file_name.split('/')[2]  #Extracting the file name from the path   
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

    sp.io.savemat('/home/amith/Desktop/oculus/PostInternship/Output_files/data_before_bandpass.mat',final_data) # Save the dictionary to a mat file.

    i=0

    for key,value in final_data.items():
        if(key[2]!='1'):
            for trial in range(0,27):
                for channel in range(0,4):
                    value[trial][channel]=butter_bandpass_filter(value[trial][channel],0.5,30,256,5)
        else:
            for trial in range(0,27):
                for channel in range(0,4):
                    value[trial][channel][0:256]=butter_bandpass_filter(value[trial][channel][0:256],0.5,30,256,5)
                    value[trial][channel][256:512]=butter_bandpass_filter(value[trial][channel][256:512],0.5,30,256,5)

    sp.io.savemat('/home/amith/Desktop/oculus/PostInternship/Output_files/completely_preprocessed_data.mat',final_data)




















def psd():
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
                    io.savemat('Output_files/'+outMatrix[0],s)
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
                    io.savemat('Output_files/'+outMatrix[1],s)
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
                    io.savemat('Output_files/'+outMatrix[2],s)
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
                    io.savemat('Output_files/'+outMatrix[3],s)
                    flagd=0


















def Classifier_RFC():

    matr = scipy.io.loadmat('Output_files/Right.mat')
    matl = scipy.io.loadmat('Output_files/Left.mat')
    matu = scipy.io.loadmat('Output_files/Up.mat')
    matd = scipy.io.loadmat('Output_files/Down.mat')


    PSD_feature = np.ndarray(shape=(432,12))
    PSD_feature = np.vstack((matl['PSD'],matr['PSD'],matu['PSD'],matd['PSD']))

    l=[]
    #108 labels for each direction
    for i in range(0,108):
        l.append(int(0))
    for i in range(0,108):
        l.append(int(1))
    for i in range(0,108):
        l.append(int(2))
    for i in range(0,108):
        l.append(int(3))

    l=np.array(l)

    X = PSD_feature

    X1=preprocessing.MinMaxScaler()
    y=l
    score=[]
    print('Feature : PSD_feature')
    print('4 Class problem')

    
    score=[]
    clf=RandomForestClassifier()#100,alpha=0.0001,max_iter=1350)
    skf = StratifiedKFold(n_splits=5, random_state=None)
    # X is the feature set and y is the target
    for train_index, val_index in skf.split(X,y): 
        X_train, X_test = X[train_index], X[val_index] 
        y_train, y_test = y[train_index], y[val_index]
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        score.append(confidence)
    #print(score)
    print('RandomForestClassifier:',statistics.mean(score))