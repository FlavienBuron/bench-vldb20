import math
import random

import numpy as np


NB_FEATURES = 1

class DataLoader():
    def __init__(self, input, isNormal=False):
        """
        input = matrix representing the dataset (numpy array)
        """
        fileNames=[]
        self.fileNames = fileNames
        mean = [0.0]*NB_FEATURES
        meancount = [0]*NB_FEATURES
        x = []
        times = []

        for row in input.transpose():
            # transpose input so each timeseries are rows

            t = 0
            totalData = []
            t_times = []
            for value in row:
                timestamp = t
                t_times.append(float(timestamp))
                totalData.append([float(value)])
                if not np.isnan(value):
                    mean[0] += float(value)
                    meancount[0] += 1
                t += 1
            x.append(totalData)
            times.append(t_times)
        
        self.x = x
        self.times = times

        for i in range(len(mean)):
            if meancount[i] != 0:
                mean[i] = mean[i]/meancount[i]
        self.mean = mean

        m = []

        self.std = [0.0]*NB_FEATURES
        for ts in self.x:
            ts_m = []
            for values in ts:
                v_m = ~np.isnan(values)
                if isNormal:
                    for i in range(len(values)):
                        if not np.isnan(value[i]):
                            self.std[i] += (value[i] - mean[i])**2
                ts_m.append(v_m)
            m.append(ts_m)
        
        for i in range(len(self.std)):
            self.std[i] = math.sqrt(1.0/(meancount[i-1])*self.std[i])

        self.isNormal = isNormal
        self.normalization(isNormal)

        x_lengths = [] 
        deltaPre = [] #time difference
        lastvalues= [] # last observed value
        deltaSub = [] # time difference in backward direction
        subvalues = [] # last value in backward direction ?

        for idx in range(len(x)):
            time_serie = self.x[idx]
            time = self.times[idx]
            x_lengths.append(len(time_serie))

            ts_deltaPre = []
            ts_lastvalues = []

            ts_deltaSub = []
            ts_subvalues = []

            ts_m = m[idx]

            for i in range(len(time_serie)):
                t_deltaSub = [0.0]*NB_FEATURES
                t_lastvalue = [0.0]*NB_FEATURES
                ts_deltaPre.append(t_deltaSub)
                ts_lastvalues.append(t_lastvalue)

                if i == 0:
                    for j in range(len(time_serie[i])):
                        ts_lastvalues[i][j]=0.0 if ts_m[i][j]==0 else time_serie[i][j]
                    continue
                
                for j in range(len(time_serie[i])):
                    if ts_m[i-1][j]==1: # if previous value was not a missing value
                        ts_deltaPre[i][j]=time[i]-time[i-1]
                    else: # else was missing
                        ts_deltaPre[i][j] = time[i]-time[i-1] + ts_deltaPre[i-1][j]

                    if ts_m[i][j]==1:
                        ts_lastvalues[i][j] = time_serie[i][j]
                    else:
                        ts_lastvalues[i][j] = ts_lastvalues[i-1][j]
            

            for i in range(len(time_serie)):
                t_deltaSub = [0.0]*NB_FEATURES
                t_subvalues = [0.0]*NB_FEATURES
                ts_deltaSub.append(t_deltaSub)
                ts_subvalues.append(t_subvalues)
            # go in backward direction
            for i in range(len(time_serie)-1, -1, -1):
                if i == len(time_serie)-1:
                    for j in range(len(time_serie[i])):
                        ts_subvalues[i][j]=0.0 if ts_m[i][j]==0 else time_serie[i][j]
                    continue
                for j in range(len(time_serie[i])):
                    if ts_m[i+1][j] == 1:
                        ts_deltaSub[i][j] = time[i+1] - time[i]
                    else:
                        ts_deltaSub[i][j] = time[i+1] - time[i] + ts_deltaSub[i+1][j]

                if ts_m[i][j] == 1:
                    ts_subvalues[i][j]=time_serie[i][j]
                else:
                    ts_subvalues[i][j] = ts_subvalues[i+1][j]

            deltaPre.append(ts_deltaPre)
            lastvalues.append(ts_lastvalues)
            deltaSub.append(ts_deltaSub)
            subvalues.append(ts_subvalues)

        self.m=m
        self.deltaPre = deltaPre
        self.lastvalues = lastvalues
        self.deltaSub = deltaSub
        self.subvalues = subvalues
        self.x_lengths = x_lengths
        self.maxLength = max(x_lengths)
        self.x = np.nan_to_num(x)
        self.times = times

        print("max_length is: " + str(self.maxLength))

    def normalization(self,isNormal):
        if not isNormal:
            return
        for ts in self.x:
            for value in ts:
                if value != np.nan:
                    if self.std==0:
                        value=0.0
                    else:
                        value=1.0/self.std*(value-self.mean)

    def nextBatch(self):
        i = 1
        while i*self.batchSize <= len(self.x):
            x = []
            y = [] # no labels here
            m = []
            deltaPre = []
            x_lengths = []
            lastvalues = []
            deltaSub = []
            subvalues = []
            imputed_deltapre = []
            imputed_m = []
            imputed_deltasub = []
            mean = self.mean
            files = []

            for j in range((i-1)*self.batchSize, i*self.batchSize):
                # files.append(self.fileNames[j])
                x.append(self.x[j])
                # y.append(self.y[j])
                m.append(self.m[j])
                deltaPre.append(self.deltaPre[j])
                deltaSub.append(self.deltaSub[j])
                x_lengths.append(self.x_lengths[j])
                lastvalues.append(self.lastvalues[j])
                subvalues.append(self.subvalues[j])
                jj = j-(i-1)*self.batchSize

                while len(x[jj]) < self.maxLength:
                    t1 = [0.0]*NB_FEATURES
                    x[jj].append(t1)
                    t2 = [0]*NB_FEATURES
                    m[jj].append(t2)
                    t3 = [0.0]*NB_FEATURES
                    deltaPre[jj].append(t3)
                    t4 = [0.0]*NB_FEATURES
                    lastvalues[jj].append(t4)
                    t5 = [0.0]*NB_FEATURES
                    deltaSub[jj].append(t5)
                    t6 = [0.0]*NB_FEATURES
                    subvalues[jj].append(t6)

            for j in range((i-1)*self.batchSize, i*self.batchSize):
                one_imputed_deltapre = []
                one_imputed_deltasub = []
                one_G_m = []
                for h in range(0, self.x_lengths[j]):
                    if h == 0:
                        one_f_time=[0.0]*NB_FEATURES
                        one_imputed_deltapre.append(one_f_time)
                        try:
                            one_sub=[self.times[j][h+1]-self.times[j][h]]*NB_FEATURES
                        except:
                            print("error: "+str(h)+" "+str(len(self.times[j])))
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*NB_FEATURES
                        one_G_m.append(one_f_g_m)
                    elif h==self.x_lengths[j]-1:
                        one_f_time=[self.times[j][h]-self.times[j][h-1]]*NB_FEATURES
                        one_imputed_deltapre.append(one_f_time)
                        one_sub=[0.0]*NB_FEATURES
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*NB_FEATURES
                        one_G_m.append(one_f_g_m)
                    else:
                        one_f_time=[self.times[j][h]-self.times[j][h-1]]*NB_FEATURES
                        one_imputed_deltapre.append(one_f_time)
                        one_sub=[self.times[j][h+1]-self.times[j][h]]*NB_FEATURES
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*NB_FEATURES
                        one_G_m.append(one_f_g_m)
                while len(one_imputed_deltapre)<self.maxLength:
                    one_f_time=[0.0]*NB_FEATURES
                    one_imputed_deltapre.append(one_f_time)
                    one_sub=[0.0]*NB_FEATURES
                    one_imputed_deltasub.append(one_sub)
                    one_f_g_m=[0.0]*NB_FEATURES
                    one_G_m.append(one_f_g_m)
                imputed_deltapre.append(one_imputed_deltapre)
                imputed_deltasub.append(one_imputed_deltasub)
                imputed_m.append(one_G_m)
            i+=1
            if self.isNormal:
                yield  x,y,[0.0]*NB_FEATURES,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
            else:
                yield  x,y,mean,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub

    def shuffle(self,batchSize=32,isShuffle=False):
        self.batchSize=batchSize
        if isShuffle:
            c = list(zip(self.x,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.fileNames,self.times,self.deltaSub,self.subvalues))
            random.shuffle(c)
            self.x,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.fileNames,self.times,self.deltaSub,self.subvalues=zip(*c)
            print("shuffled")
