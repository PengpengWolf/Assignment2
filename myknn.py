# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:50:01 2020

@author: irisy
"""
import numpy as np
import time
import matplotlib.pyplot as plt 
import random

#def seperateTT(dataset,c,n_timesteps):
#    training=[]
#    test=[]
#   for i in range(len(dataset)):
#        if i<c :
#            training.append(dataset[i])
#        elif (i>=n_timesteps and i<(n_timesteps+c)):
#            training.append(dataset[i])
#        elif (i>=2*n_timesteps and i<(2*n_timesteps+c)):
#            training.append(dataset[i])
#        elif (i>=3*n_timesteps and i<(3*n_timesteps+c)):
#            training.append(dataset[i])
#        elif (i>=4*n_timesteps and i<(4*n_timesteps+c)):
#            training.append(dataset[i])
#        else:
#            test.append(dataset[i])
#    return training, test

def seperateTT(dataset,c,n_timesteps):
    training=[]
    test=[]
    training=random.sample(dataset, c)
    for element in dataset:
        if element not in training:
            test.append(element)
        
    return training, test

def euclDistance(dataset, center):
    sum=0
    #for i in range(len(dataset)):
    #sum+=((dataset[i][0]-center[0])**2+(dataset[i][1]-center[1])**2)   
    sum+=((dataset[0]-center[0])**2+(dataset[1]-center[1])**2)     
    sum=np.sqrt(sum)
    return sum

def getdistance(testset,trainset):
    dis_1=[]
    dis_2=[]
    dis_tot=[]
    test_point=[]
    train_point=[]
    for i in range(len(testset)):
        test_point=testset[i]
        dis_1=[]
        dis_2=[]
        for j in range(len(trainset)):
            train_point=trainset[j]
            dis_1=euclDistance(test_point[0],train_point[0])
            dis_2.append([dis_1,train_point[1]])
            #print('dis_1',dis_1)
            #print('dis_2',dis_2)
        dis_tot.append([dis_2,testset[i][0],testset[i][1]])
        #print()
    return dis_tot

def takefirst(elem):
    return elem[0]
def takethird(elem):
    return elem[2]

def voteKNN(distance,test_num):
    num=0
    num0=0
    num1=0
    num2=0
    num3=0
    num4=0
    #print('try')
    #distance.sort(key=takefirst)
    #for i in range(len(distance)):
    #    distance[i][0].sort()
    distance[0].sort(key=takefirst)
    test_example=distance[0][0:test_num]
    for i in range(len(test_example)):
        if test_example[i][1]==0:
            num0+=1
        elif test_example[i][1]==1:
            num1+=1
        elif test_example[i][1]==2:
            num2+=1
        elif test_example[i][1]==3:
            num3+=1
        elif test_example[i][1]==4:
            num4+=1
    num=[num0,num1,num2,num3,num4]
    index=num.index(max(num))

    return index

def compareResult(dis_tot,test_num):
    dis_tot_tag=[]
    non_correct=0
    yes_correct=0
    for i in range(len(dis_tot)):
        new_Index=voteKNN(dis_tot[i],test_num)
        dis_tot_tag.append([dis_tot[i][0],dis_tot[i][1],dis_tot[i][2],new_Index,'None'])
    for j in range(len(dis_tot_tag)):
        if dis_tot_tag[j][2] != dis_tot_tag[j][3]:
            non_correct+=1
            dis_tot_tag[j][4]='False'
        else:
            yes_correct+=1
            dis_tot_tag[j][4]='True'
        
    return dis_tot_tag,non_correct,yes_correct
