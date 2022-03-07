from code import interact
from copy import deepcopy
from ctypes import util
import enum
from tty import CC
from typing import final
from unittest import result
import env
import numpy as np
from matplotlib import pyplot as plt 

# numofVehicle = 6
# vehicleLocation = [(0,0.001),(0,0.001),(0,0.001),(0,0.001),(0,0.001),(0,0.001)]
# vehicleCPU = [2,2,2,2,2,2]
# vehicleTaskprofit = [(10,1,5),(10,2,10),(10,1.5,7.5),(10,1,5),(10,2,10),(10,1.5,7.5)]
numofVehicle = 6
vehicleLocation = [(0,0.001),(0,0.001),(0,0.001),(0,0.001),(0,0.001),(0,0.001)]
vehicleCPU = [2,2,2,2,2,2]
vehicleTmax = [2.2,2.64,4.4,1.8,3.3,0.98]
# T_Best:1,1.2,2,1,1.5,0.7
vehicleTaskprofit = [(10,1,2.2),(8,1.5,2.64),(10,2,4.4),(5,2,1.8),(10,1.5,3.3),(7,1,0.98)]  
vehicleFavourateTime = [0.7,0.84,1.4,0.8,1.05,0.63]  ## 这个不是车辆最喜欢的，这个就是best*0.7


### TODO:考虑再分别定义系统希望每个任务计算的时间间隔(这里注意是  时间间隔 )
expectTime = np.zeros(numofVehicle)
lamda = 0 ##  总任务到达间隔 1.25s 来一个任务
for veh in range(numofVehicle):
    expectTime[veh] = vehicleFavourateTime[veh-1] + expectTime[veh-1] if veh > 0 else 0
    lamda += vehicleFavourateTime[veh]
### ☝同时定义了 到达率 和 各任务起始时间


mec = env.mecnode(numofVehicle,8,(0,0),vehicleLocation=vehicleLocation,vehicleCPU=vehicleCPU,vehicleTaskprofit=vehicleTaskprofit,lamda=lamda,sublamda = expectTime)

actionCandicate = np.linspace(0,1,500)

#### 构建一个最初的past  ################################
actionPast = []
for veh in range(numofVehicle):
    actionPast.append(0.05*vehicleTaskprofit[veh][0])


cCPU = 0
iteratorTime = 50
priceUtilityArray = []
actualUtilityArray = []
UtilitySumArray = []
actionArray = []
# for iter in range(iteratorTime):
cCPUCandicata = np.linspace(0,1,30)
mec.reset()
actioninThisIter = np.zeros(numofVehicle)
for iter,cCPU in enumerate(cCPUCandicata):
    ## 对每个车求最大的action
   
    for veh in range(numofVehicle):
        uveh = []
        for action in actionCandicate:
            actionPast[veh] = action*vehicleTaskprofit[veh][0]
            # u,_ = mec.utility(actionPast)
            _,u = mec.utility(actionPast)
            uveh.append(u)
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
    actionArray.append(deepcopy(actioninThisIter))
    UtilitySumArray.append(sum(actualUtility))
    priceUtilityArray.append(np.var(np.array(priceUtility)))
    actualUtilityArray.append(np.var(np.array(actualUtility)))
    print("第{}次迭代，三者的效用为：{},效用方差为：{}".format(iter,actualUtility,np.var(np.array(actualUtility))))
    print("此时运行到第{}s,当前的idletime={}".format(mec.index,mec.idletime))
    print("--------------------------------------------------------------------------------")
    mec.setCCPU(cCPU)

# print("result:{}".format(result))


for index,utility in enumerate(UtilitySumArray):
    UtilitySumArray[index] = UtilitySumArray[index-1] + utility if index > 0 else utility



fig, ax = plt.subplots() # 创建图实例



ax.plot(cCPUCandicata, np.array(actionArray)) # 作y1 = x 图，并标记此线名为linear
# ax.plot(cCPUCandicata, actualUtilityArray, label='actual') #作y2 = x^2 图，并标记此线名为quadratic

ax.set_xlabel('x label') #设置x轴名称 x label
ax.set_ylabel('y label') #设置y轴名称 y label
ax.set_title('Simple Plot') #设置图名为Simple Plot
ax.legend() #自动检测要在图例中显示的元素，并且显示

plt.show() #图形可视化




