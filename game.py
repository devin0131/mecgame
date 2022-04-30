from code import interact
from copy import deepcopy
from ctypes import util
import enum
from tkinter import font
from tty import CC
from typing import final
from unittest import result
import env
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
# numofVehicle = 6
# vehicleLocation = [(0,0.001),(0,0.001),(0,0.001),(0,0.001),(0,0.001),(0,0.001)]
# vehicleCPU = [2,2,2,2,2,2]
# vehicleTaskprofit = [(10,1,5),(10,2,10),(10,1.5,7.5),(10,1,5),(10,2,10),(10,1.5,7.5)]
numofVehicle = 6
vehicleLocation = [(0,0.001),(0,0.001),(0,0.001),(0,0.001),(0,0.001),(0,0.001)]
vehicleCPU = [2,2,2,2,2,2]
vehicleTmax = [2.2,2.64,4.4,1.8,3.3,0.98]
# T_Best:1,1.2,2,1,1.5,0.7
vehicleTaskprofit = [(10,1,3),(8,1.5,3),(10,2,5),(5,2,2.5),(10,1.5,4),(7,1,1.5)]
vehicleFavourateTime = [0.7,0.84,1.4,0.8,1.05,0.63]  ## 这个不是车辆最喜欢的，这个就是best*0.7
wantedVar = 0.1

### TODO:考虑再分别定义系统希望每个任务计算的时间间隔(这里注意是  时间间隔 )
expectTime = np.zeros(numofVehicle)
lamda = 0 ##  总任务到达间隔 1.25s 来一个任务
for veh in range(numofVehicle):
    expectTime[veh] =  ( vehicleFavourateTime[veh-1] + expectTime[veh-1] )  if veh > 0 else 0
    lamda += vehicleFavourateTime[veh]
### ☝同时定义了 到达率 和 各任务起始时间


mec = env.mecnode(numofVehicle,8,(0,0),vehicleLocation=vehicleLocation,vehicleCPU=vehicleCPU,vehicleTaskprofit=vehicleTaskprofit,lamda=lamda,sublamda = expectTime)

actionCandicate = np.linspace(0,1,500)

#### 构建一个最初的past  ################################
actionPast = []
for veh in range(numofVehicle):
    actionPast.append(0.05*vehicleTaskprofit[veh][0])


iteratorTime = 20
priceUtilityArray = []
actualUtilityArray = []
UtilitySumArray = []
actionArray = []
utilityArray = []
# utilityVarArray = []
# for iter in range(iteratorTime):

cCPUCandicata = np.linspace(0,1,iteratorTime)
mec.reset()
actioninThisIter = np.zeros(numofVehicle)
for iter,cCPU in enumerate(cCPUCandicata):
    ## 对每个车求最大的action
    for veh in range(numofVehicle):
        uveh = []
        for action in actionCandicate:
            actionPast[veh] = action*vehicleTaskprofit[veh][0]
            # u,_ = mec.utility(actionPast)
            pu,u = mec.utility(actionPast)
            uveh.append(pu)
        uveh = np.array(uveh)[:,veh]
        uvehMaxIndex = uveh.argmax()
        actioninThisIter[veh] = actionCandicate[uvehMaxIndex]
        print("第{}次迭代，车辆{}的效用:最佳决策：{}\n".format(iter,veh,uvehMaxIndex,uveh))
        actionPast[veh] = actionCandicate[uvehMaxIndex]*vehicleTaskprofit[veh][0]
        
        
    
    priceUtility,actualUtility = mec.utility(actionPast)
    ### 采取各节点所认为的最佳决策之后,更新系统的idletime值和index值(index可以代表经历了多少个时刻)
    mec.idletime = mec.idletime_last
    mec.index += mec.lamda
    ### 保存实验数据
    utilityArray.append(actualUtility)
    actionArray.append(deepcopy(actioninThisIter))
    UtilitySumArray.append(sum(actualUtility))
    priceUtilityArray.append(np.var(np.array(priceUtility)))
    actualUtilityArray.append(np.var(np.array(actualUtility)))
    print("第{}次迭代，三者的效用为：{},效用方差为：{}".format(iter,actualUtility,np.var(np.array(actualUtility))))
    print("此时运行到第{}s,当前的idletime={}".format(mec.index,mec.idletime))
    print("--------------------------------------------------------------------------------")
    if(np.var(np.array(actualUtility)) > wantedVar and iter>10):
        mec.setCCPU(actualUtility)
        # pass
    else:
        print("不用set了")

# print("result:{}".format(result))


for index,utility in enumerate(UtilitySumArray):
    UtilitySumArray[index] = UtilitySumArray[index-1] + utility if index > 0 else utility


line_shape = ["*-","^-",".-","s-","D-","1-"]
x = range(1,iteratorTime+1)
legend = ["vehicle1","vehicle2","vehicle3","vehicle4","vehicle5","vehicle6"]
for index in range(numofVehicle):
    plt.plot(x,np.array(utilityArray)[:,index],line_shape[index],label=legend[index],linewidth=1.3,markersize=8)
fontsize = 11
plt.grid()
plt.legend(fontsize=fontsize)
plt.xticks(x,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim(min(x),max(x))
# plt.ylim(0.2,1)
plt.xlabel("Iterator",fontsize=fontsize + 3)
plt.ylabel("Utility",fontsize = fontsize+3)
# plt.show()

plt.figure()
for index in range(numofVehicle):
    plt.plot(x,np.array(actionArray)[:,index],line_shape[index],label=legend[index],linewidth=1.3,markersize=8)
plt.grid()
plt.legend(fontsize=fontsize)
plt.xticks(x,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim(min(x),max(x))
# plt.ylim(0.2,1)
plt.xlabel("Iterator",fontsize=fontsize + 3)
plt.ylabel("Action",fontsize = fontsize+3)

plt.show() #图形可视化




