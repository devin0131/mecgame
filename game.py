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
mecCPU = 8
discountFactor = 0.85
phi = -0.1
# 车辆的位置坐标
vehicleLocation = [(0,0.001),(0,0.001),(0,0.001),(0,0.001),(0,0.001),(0,0.001)]
# 车辆的CPU算力
vehicleCPU = [1.5,2,1.8,2,1.7,2]
# 任务文件大小，任务复杂度，任务时延约束
vehicleTaskprofit = [[7,1,1.5],[10,1,3],[8,1.5,3],[10,1.5,4],[6,2,2.5],[10,2,5]]
# 不考虑排队时延下的最佳计算时间
vehicleBestTime = np.zeros(numofVehicle)
vehicleDiscountTime = np.zeros(numofVehicle)  ## 这个不是车辆最喜欢的，这个就是best*0.7
vehicleArriveTime = np.zeros(numofVehicle)
vehicleCost = np.zeros((numofVehicle,2))
lamda = 0 ## 所有任务到达间隔

for vec in range(numofVehicle):
    ## 每个任务最喜欢的处理时延
    vehicleBestTime[vec] = ( vehicleTaskprofit[vec][0]*vehicleTaskprofit[vec][1] ) / (vehicleCPU[vec] + mecCPU)
    ### DiscountTime 这个变量可以用来做车辆的到达时间 后边的乘数是服务器不想按找最佳卸载
    vehicleDiscountTime[vec] = vehicleBestTime[vec] * discountFactor
    ## 任务的时延约束比Best 多出来一点儿
    vehicleTaskprofit[vec][2] = vehicleBestTime[vec]  * (1 + (((1 - discountFactor) * mecCPU) / vehicleCPU[vec]))
    ## 每个任务在一个时隙内的到达时间
    vehicleArriveTime[vec] =  ( vehicleDiscountTime[vec-1] + vehicleArriveTime[vec-1] )  if vec > 0 else 0
    ## 所有任务的到达时间
    lamda += vehicleDiscountTime[vec]
    vehicleCost[vec][0] = (vehicleTaskprofit[vec][0] * vehicleTaskprofit[vec][1] / vehicleCPU[vec]) - (vehicleDiscountTime[vec]*(1+(mecCPU/vehicleCPU[vec])))
    vehicleCost[vec][1] = 1 + (phi * (1 + (mecCPU / vehicleCPU[vec])))






""" wantedVar是为了定义想要的所有车辆方差的最大值
"""
wantedVar = 0.01

print("Ready to starting...")
print("vehicleLocation:{}".format(vehicleLocation))
print("vehicleCPU:{}".format(vehicleCPU))
print("vehicleTaskprofit:{}".format(vehicleTaskprofit))
print("vehicleBestTime:{}".format(vehicleBestTime))
print("vehicleDiscountTime:{}".format(vehicleDiscountTime))
print("vehicleArriveTime:{}".format(vehicleArriveTime))


mec = env.mecnode(numofVehicle,mecCPU,(0,0),vehicleLocation=vehicleLocation,vehicleCPU=vehicleCPU,vehicleTaskprofit=vehicleTaskprofit,lamda=lamda,sublamda = vehicleArriveTime,vehicleCost = vehicleCost)

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
        print("第{}次迭代，车辆{}的效用:最佳决策：{}".format(iter,veh,uvehMaxIndex,uveh))
        actionPast[veh] = actionCandicate[uvehMaxIndex]*vehicleTaskprofit[veh][0]

    ### 采取各节点所认为的最佳决策之后,更新系统的idletime值和index值(index可以代表经历了多少个时刻)
    priceUtility, actualUtility = mec.utility(actionPast,showlog = True)
    mec.idletime = mec.idletime_last
    mec.timeNow += mec.lamda
    ### 保存实验数据
    utilityArray.append(actualUtility)
    actionArray.append(deepcopy(actioninThisIter))
    UtilitySumArray.append(sum(actualUtility))
    priceUtilityArray.append(np.var(np.array(priceUtility)))
    actualUtilityArray.append(np.var(np.array(actualUtility)))
    print("第{}次迭代，\n效用为：\n{},\n效用方差为：\n{}".format(iter,actualUtility,np.var(np.array(actualUtility))))
    print("此时运行到第{}s,当前的idletime={}".format(mec.timeNow,mec.idletime))
    print("--------------------------------------------------------------------------------")
    ## 判定条件
    if(np.var(np.array(actualUtility)) > wantedVar and iter>10):
        ## mec.setCCPU2(actualUtility)
        mec.setCCPU3(True)
    else:
        print("no set")

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




