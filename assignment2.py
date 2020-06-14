# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:53:04 2020

@author: irisy
"""


import os
import numpy as np
import pandas as pd
import tempfile
import matplotlib
import matplotlib.pyplot as plt
import random

import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
from pandapower.plotting.plotly import simple_plotly
import mykmeans as mkm
import myknn as mknn
def timeseries(output_dir):
    # 1. create test net
    net = create_net()

    # 2. create (random) data source
    n_timesteps = 60
    profiles, ds = create_data_source(n_timesteps)
    #print(ds)
    #print(profiles)
    # 3. create controllers (to control P values of the load and the sgen)
    create_controllers(net, ds)
    #print(net)
    #print(ds)
    #print(profiles)
    #pp.plotting.simple_plot(net)
    # time steps to be calculated. Could also be a list with non-consecutive time steps
    time_steps = range(0, n_timesteps)
   # print(time_steps)
    
    # 4. the output writer with the desired results to be stored to files.
    ow = create_output_writer(net, time_steps, output_dir=output_dir)

    # 5. the main time series function
    run_timeseries(net, time_steps)

def create_net(state='NS'):
    net = pp.create_empty_network()
    pp.set_user_pf_options(net, init_vm_pu = "flat", init_va_degree = "dc", calculate_voltage_angles=True)
       
    b0 = pp.create_bus(net, 110)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    b4 = pp.create_bus(net, 110)
    b5 = pp.create_bus(net, 110)
    b6 = pp.create_bus(net, 110)
    b7 = pp.create_bus(net, 110)
    b8 = pp.create_bus(net, 110)
    
    pp.create_ext_grid(net, b0)
    
    pp.create_line(net, b0, b3, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b3, b4, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b3, b8, 10, "149-AL1/24-ST1A 110.0")
    if state in ['NS','HL','LL','GD']:
        pp.create_line(net, b4, b5, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b5, b2, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b8, b7, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b5, b6, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b1, b7, 10, "149-AL1/24-ST1A 110.0")
    pp.create_line(net, b7, b6, 10, "149-AL1/24-ST1A 110.0")
    
    if state == 'HL':
        pp.create_load(net, b4, p_mw=90*1.1, q_mvar=30*1.1, name='load1')
        pp.create_load(net, b6, p_mw=100*1.1, q_mvar=35*1.1, name='load2')
        pp.create_load(net, b8, p_mw=125*1.1, q_mvar=50*1.1, name='load3')
    
    if state == 'LL':
        pp.create_load(net, b4, p_mw=90*0.9, q_mvar=30*0.9, name='load1')
        pp.create_load(net, b6, p_mw=100*0.9, q_mvar=35*0.9, name='load2')
        pp.create_load(net, b8, p_mw=125*0.9, q_mvar=50*0.9, name='load3')   
    else: 
        pp.create_load(net, b4, p_mw=90, q_mvar=30, name='load1')
        pp.create_load(net, b6, p_mw=100, q_mvar=35, name='load2')
        pp.create_load(net, b8, p_mw=125, q_mvar=50, name='load3')
        
    pp.create_gen(net, b0, p_mw=0, vm_pu=1.0, name='gen1', slack=True)    
    pp.create_sgen(net, b1, p_mw=163, q_mvar=0, name='sgen1')
    if state in ['NS','HL','LL','LD']:
        pp.create_sgen(net, b2, p_mw=85, q_mvar=0, name='sgen2')
    
    return net

def create_data_source(n_timesteps=60,state='NS'):
    profiles = pd.DataFrame()
    if state=='HL':
        profiles['load1_p'] = np.random.random(n_timesteps) * 90*1.1
        profiles['load2_p'] = np.random.random(n_timesteps) * 100*1.1
        profiles['load3_p'] = np.random.random(n_timesteps) * 125*1.1
    if state=='LL':
        profiles['load1_p'] = np.random.random(n_timesteps) * 90*0.9
        profiles['load2_p'] = np.random.random(n_timesteps) * 100*0.9
        profiles['load3_p'] = np.random.random(n_timesteps) * 125*0.9        
    if state=='NS':
        profiles['load1_p'] = np.random.random(n_timesteps) * 90
        profiles['load2_p'] = np.random.random(n_timesteps) * 100
        profiles['load3_p'] = np.random.random(n_timesteps) * 125        
        
        profiles['sgen1_p'] = np.random.random(n_timesteps) * 163
    if state in ['NS','HL','LL','LD']:
        profiles['sgen2_p'] = np.random.random(n_timesteps) * 85 
  
    ds = DFData(profiles)
    return profiles, ds

def create_controllers(net, ds,state='NS'):
    ConstControl(net, element='load', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["load1_p"])
    ConstControl(net, element='load', variable='p_mw', element_index=[1],
                 data_source=ds, profile_name=["load2_p"])
    ConstControl(net, element='load', variable='p_mw', element_index=[2],
                 data_source=ds, profile_name=["load3_p"])    
    ConstControl(net, element='sgen', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["sgen1_p"])
    if state in ['NS','HL','LL','LD']:
        ConstControl(net, element='sgen', variable='p_mw', element_index=[1],
                    data_source=ds, profile_name=["sgen2_p"])

    
def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop
    #ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
   # ow.log_variable('res_line', 'loading_percent')
    #ow.log_variable('res_line', 'i_ka')
    return ow
output_dir_HL = os.path.join(tempfile.gettempdir(), "time_series_example_HL")
output_dir_LL = os.path.join(tempfile.gettempdir(), "time_series_example_LL")
output_dir_GD = os.path.join(tempfile.gettempdir(), "time_series_example_GD")
output_dir_LD = os.path.join(tempfile.gettempdir(), "time_series_example_LD")
output_dir_NS = os.path.join(tempfile.gettempdir(), "time_series_example_NS")

output_dir_NS = os.path.join(tempfile.gettempdir(), "time_series_vm")
print("Results can be found in your local temp folder: {}".format(output_dir_HL))
if not os.path.exists(output_dir_HL):
    os.mkdir(output_dir_HL)
timeseries(output_dir_HL)
timeseries(output_dir_LL)
timeseries(output_dir_GD)
timeseries(output_dir_LD)
timeseries(output_dir_NS)

vm_pu_file = os.path.join(output_dir_HL, "res_bus", "vm_pu.xls")
vm_pu_1 = pd.read_excel(vm_pu_file)
angle_deg_file = os.path.join(output_dir_HL, "res_bus", "va_degree.xls")
angle_deg_1 = pd.read_excel(angle_deg_file)



vm_pu_file = os.path.join(output_dir_LL, "res_bus", "vm_pu.xls")
vm_pu_2 = pd.read_excel(vm_pu_file)
angle_deg_file = os.path.join(output_dir_LL, "res_bus", "va_degree.xls")
angle_deg_2 = pd.read_excel(angle_deg_file)
temp_vm1=np.vstack((vm_pu_1,vm_pu_2))
temp_ang1=np.vstack((angle_deg_1,angle_deg_2))

vm_pu_file = os.path.join(output_dir_GD, "res_bus", "vm_pu.xls")
vm_pu_3 = pd.read_excel(vm_pu_file)
angle_deg_file = os.path.join(output_dir_GD, "res_bus", "va_degree.xls")
angle_deg_3 = pd.read_excel(angle_deg_file)
temp_vm2=np.vstack((temp_vm1,vm_pu_3))
temp_ang2=np.vstack((temp_ang1,angle_deg_3))

vm_pu_file = os.path.join(output_dir_LD, "res_bus", "vm_pu.xls")
vm_pu_4 = pd.read_excel(vm_pu_file)
angle_deg_file = os.path.join(output_dir_LD, "res_bus", "va_degree.xls")
angle_deg_4 = pd.read_excel(angle_deg_file)
temp_vm3=np.vstack((temp_vm2,vm_pu_4))
temp_ang3=np.vstack((temp_ang2,angle_deg_4))

vm_pu_file = os.path.join(output_dir_NS, "res_bus", "vm_pu.xls")
vm_pu_5 = pd.read_excel(vm_pu_file)
angle_deg_file = os.path.join(output_dir_NS, "res_bus", "va_degree.xls")
angle_deg_5 = pd.read_excel(angle_deg_file)
vm_pu_tot=np.vstack((temp_vm3,vm_pu_5))
angle_deg_tot=np.vstack((temp_ang3,angle_deg_5))

###K-means
bus_num=2
data_bus=[]
dataset=[]
data_bus.append([vm_pu_tot[:,bus_num],angle_deg_tot[:,bus_num]])
bus_var=np.var(data_bus[0][1])#calcilate the variation of the angle value
data_bus[0][1]/=bus_var#normalize the angle value
for i in range(len(data_bus[0][0])):#reshape the values,so that every element contains one vol and its corresponding angle.
    dataset.append([data_bus[0][0][i],data_bus[0][1][i]])

dim = np.shape(dataset)

k=5#innitailize cluter number
maxiter=300
#toler=0.001
centroids, clusterAssment=mkm.kmeans(dataset,k,maxiter)
dataset_tag=mkm.showCluster(dataset, k, centroids, clusterAssment)
#plt.title('Kmeans')

###KNN
n_timesteps=60
c=int(0.78*n_timesteps*5)


#training_HL, training_LL,training_GD,training_LD,training_NS, test_HL, test_LL, test_GD, test_LD, test_NS=mknn.seperateTT(dataset,c,n_timesteps)
training, test=mknn.seperateTT(dataset_tag,c,n_timesteps)
print('The data number in the training set is :', len(training))
print('The data number in the test set is :', len(test))

test_num=9#innitailize cluter number
#vote for HL status


index=[]
non_correct=0
yes_correct=0
dis=mknn.getdistance(test,training)

dis_tag,non_correct,yes_correct=mknn.compareResult(dis,test_num)
print('Non correct:',non_correct,'\n','Correct:',yes_correct,'\n','Rate of accuracy:',yes_correct/(len(test)))


