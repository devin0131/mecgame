from asyncio import Task, tasks
from dis import dis
from distutils.command.config import LANG_EXT
from lib2to3.pytree import Node
from termios import IEXTEN
import numpy as np
import math




def pNoise(T,deitaF):
    return 1.38*math.pow(10,-23)*(273.15+T)*deitaF

class vehiclenode:
    def __init__(self,capacityofCPU,location = (0,0),taskprofit = (1,1,1)) -> None:
        self.CapacityofCPU = capacityofCPU
        self.taskprofit = taskprofit #任务文件大小 任务时间复杂度 任务空间复杂度
        self.location = location
class mecnode:
    ######################### --初始化-- ###################################
    def __init__(self,numofVehicle,capacityofCPU,location,vehicleLocation,vehicleCPU,vehicleTaskprofit,lamda = 2,sublamda = None, vehicleCost = None) -> None:

        ## 系统参量
        self.numofVehicle = numofVehicle
        self.lamda = lamda
        self.sublamda = sublamda

        ## 信道参量
        self.L0 = 61094
        self.P_tx_v = 1
        self.P_w_r = pNoise(26,10e6)
        self.subBand = 10e6
        self.location = location
        self.alpha = 2.75
        #######

        ## 训练参量
        # self.cSpec = np.zeros(self.numofVehicle)
        self.cCpu = np.zeros(self.numofVehicle)
        self.timeNow = 0
        self.idletime = 0
        self.idletime_last = 0
        self.useComputeCost = False
        # # self.cMem = np.zeros(self.numofVehicle)
        # #######

        ## MEC能力参量
        self.CapacityofCPU = capacityofCPU  #NOTE：这里是MEC网络的控制器的主频
        ########

        ## 初始化车辆
        self.vehicleLocation = vehicleLocation
        self.vehicleCPU = vehicleCPU
        self.vehicleTaskprofit = vehicleTaskprofit
        self.vehicleCost = vehicleCost
        self.vehicle = []
        for index in range(self.numofVehicle):
            self.vehicle.append(vehiclenode(self.vehicleCPU[index],location=self.vehicleLocation[index],taskprofit = self.vehicleTaskprofit[index]))


        # self.lastAction = []
     #######################################################################
    ######################### --计算传输-- ###################################
    def distance(self,sour,dest):
        return np.linalg.norm(np.array(sour)-np.array(dest))
    def sinr(self,d_v_r):
        return (self.P_tx_v/(self.L0*(math.pow(d_v_r,self.alpha))))/(self.P_w_r)
    def R_v_v_(self,qv,d_v_r):
        rate = qv*math.log(1+self.sinr(d_v_r))/(8*math.pow(2,20))
        # print("R_v_v_:{0}".format(rate))
        return rate
     #######################################################################


    #########################  --控制系统-- ##################################
    def setCCPU(self,utility):
        slight = 0.2
        utility = np.array(utility) - np.array(utility).mean()
        utility -= utility.min()
        self.cCpu += utility*slight
        print("设置每个车CPU的价格:{}\n".format(self.cCpu))
    def setCCPU2(self,utility):
        utility = np.array(utility) - np.array(utility).mean()
        utility /= (utility.max() - utility.min())
        self.cCpu = 0.75 * np.log(utility  + 1)
        print("设置每个车CPU的价格:{}\n".format(self.cCpu))
    def setCCPU3(self, useornot):
        self.useComputeCost = useornot
    def reset(self):
        self.idletime = 0
        self.index = 0
     ########################################################################
    #########################  --计算效用-- ##################################

    ## params:action 保存卸载量
    ## output:utility1 价钱效用
    ##        utility2 实际效用
    def utility(self,action,showlog = False):
        completeQueue = []  ## 储存了每个任务的等待时间  包括排队的时间和计算的时间
        computeQueue = []
        waitQueue = []
        costQueue = []
        utility1 = []
        utility2 = []
        idletime = self.idletime
        for index in range(self.numofVehicle):
            taskload = action[index] * self.vehicle[index].taskprofit[1]
            computeTime = taskload / self.CapacityofCPU
            computeQueue.append(computeTime)
            arriveTime = self.timeNow+self.sublamda[index]
            # taskloadSum += taskload
            if(idletime <= arriveTime): ## 说明是空闲的
                waitQueue.append(0)
                if showlog:
                    print("第{}个车辆到达的时候，服务器是空闲的\n".format(index),end="\n\n")
                idletime = computeTime + arriveTime ## 更新计算完这个任务的时间
                completeQueue.append(computeTime)  ## 空闲的等待时间直接算就行了
            else:
                waitQueue.append(idletime - arriveTime)
                if showlog:
                    print("第{}个车辆到达的时候，服务器不是空闲的\nidletime:{},arriveTime:{},computeTime:{}".format(index,idletime,arriveTime,computeTime),end="\n\n")
                idletime = computeTime + idletime       ## 不是空闲的，就在当前idle的基础上累积，也是计算完这个任务的时间
                completeQueue.append(idletime - arriveTime)  ## 然后减去任务过来的时间
        self.idletime_last = idletime
        for index in range(self.numofVehicle):
            ## computeCost = self.cCpu[index] * completeQueue[index]
            if self.useComputeCost:
                computeCost = self.vehicleCost[index][0] - ( self.vehicleCost[index][1] * waitQueue[index] )
            else:
                computeCost = 0
            costQueue.append(computeCost)
            localtime = self.vehicle[index].taskprofit[1]*(self.vehicle[index].taskprofit[0] - action[index])/self.vehicle[index].CapacityofCPU
            mectime = completeQueue[index]
            utility1.append(self.vehicle[index].taskprofit[2] - max(localtime,mectime+computeCost))
            ## utility1.append(self.vehicle[index].taskprofit[2] - max(localtime,mectime) - computeCost)
            utility2.append(self.vehicle[index].taskprofit[2] - max(localtime,mectime))
        if showlog:
            print("utility1:\n{}".format(utility1))
            print("utility2:\n{}".format(utility2))
            print("costQueue:\n{}".format(costQueue))
            print("waitQueue:\n{}".format(waitQueue),end='\n\n')
        return utility1,utility2
     ########################################################################

