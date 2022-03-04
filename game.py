from code import interact
from unittest import result
import env
import numpy as np
from matplotlib import pyplot as plt 

numofVehicle = 3
vehicleLocation = [(5,0),(-5,0),(5,-5)]
vehicleCPU = [2,1.9,1.5]
vehicleTaskprofit = [(10,1,3),(10,2,4),(10,1.5,3)]
mec = env.mecnode(3,5,(0,0),vehicleLocation=vehicleLocation,vehicleCPU=vehicleCPU,vehicleTaskprofit=vehicleTaskprofit,lamda=0.8)

actionCandicate = np.linspace(0,1,21)


actionPast = []
for veh in range(numofVehicle):
    actionPast.append(0.05*vehicleTaskprofit[veh][0])


iteratorTime = 1
result = []
resultVehicle1 = []
resultVehicle2 = []
for _ in range(iteratorTime):

    ## 对每个车求最大的action
    for veh in range(numofVehicle):
        uveh = []
        for action in actionCandicate:
            actionPast[veh] = action*vehicleTaskprofit[veh][0]
            u = mec.utility(actionPast)
            uveh.append(u)
        uveh = np.array(uveh)[:,veh]
        if(veh == 0):
            resultVehicle1.append(uveh.max())
        if(veh == 1):
            resultVehicle2.append(uveh.max())
        uvehMaxIndex = uveh.argmax()
        actionPast[veh] = actionCandicate[uvehMaxIndex]*vehicleTaskprofit[veh][0]
    
    result.append(sum(mec.utility(actionPast)))

print("result:{}".format(result))
print("resultVehicle1:{}".format(resultVehicle1))
plt.plot(resultVehicle2)
plt.show()





        

        


# plt.plot(actionCandicate,result)
# plt.show()



