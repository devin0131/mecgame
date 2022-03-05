from code import interact
from tty import CC
from typing import final
from unittest import result
import env
import numpy as np
from matplotlib import pyplot as plt 

numofVehicle = 3
vehicleLocation = [(5,0),(-5,0),(5,-5)]
vehicleCPU = [2,1.9,1.5]
vehicleTaskprofit = [(10,1,2.5),(10,2,4),(10,1.5,3)]
mec = env.mecnode(3,5,(0,0),vehicleLocation=vehicleLocation,vehicleCPU=vehicleCPU,vehicleTaskprofit=vehicleTaskprofit,lamda=0.8)

actionCandicate = np.linspace(0,1,21)

#### 构建一个最初的past  ################################
actionPast = []
for veh in range(numofVehicle):
    actionPast.append(0.05*vehicleTaskprofit[veh][0])


cCPU = 0
cCPUSTEP = 0.1
iteratorTime = 50
priceUtilityArray = []
actualUtilityArray = []
# for iter in range(iteratorTime):
cCPUCandicata = np.linspace(-10,5,10000)
for cCPU in cCPUCandicata:

    ## 对每个车求最大的action
    for veh in range(numofVehicle):
        uveh = []
        for action in actionCandicate:
            actionPast[veh] = action*vehicleTaskprofit[veh][0]
            u,_ = mec.utility(actionPast)
            uveh.append(u)
        uveh = np.array(uveh)[:,veh]
        uvehMaxIndex = uveh.argmax()
        # print("第{}次迭代，车辆{}的效用:最佳决策：{}\n{}".format(iter,veh,uvehMaxIndex,uveh))
        actionPast[veh] = actionCandicate[uvehMaxIndex]*vehicleTaskprofit[veh][0]
    
    priceUtility,actualUtility = mec.utility(actionPast)
    
    # result.append(sum(finalUtility))
    # result.append(np.var(np.array(finalUtility)))
    priceUtilityArray.append(np.var(np.array(priceUtility)))
    actualUtilityArray.append(np.var(np.array(actualUtility)))
    # print("第{}次迭代，三者的效用为：{},效用方差为：{}".format(iter,finalUtility,np.var(np.array(finalUtility))))
    mec.setCCPU(cCPU)

# print("result:{}".format(result))


fig, ax = plt.subplots() # 创建图实例

ax.plot(cCPUCandicata, priceUtilityArray, label='price') # 作y1 = x 图，并标记此线名为linear
ax.plot(cCPUCandicata, actualUtilityArray, label='actual') #作y2 = x^2 图，并标记此线名为quadratic

ax.set_xlabel('x label') #设置x轴名称 x label
ax.set_ylabel('y label') #设置y轴名称 y label
ax.set_title('Simple Plot') #设置图名为Simple Plot
ax.legend() #自动检测要在图例中显示的元素，并且显示

plt.show() #图形可视化




