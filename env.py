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
    def __init__(self,numofVehicle,capacityofCPU,location,vehicleLocation,vehicleCPU,vehicleTaskprofit,lamda = 2,sublamda = None) -> None:

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
        self.cCpu = 0
        self.index = 0
        self.idletime = 0
        self.idletime_last = 0
        # # self.cMem = np.zeros(self.numofVehicle)
        # #######

        ## MEC能力参量
        self.CapacityofCPU = capacityofCPU  #NOTE：这里是MEC网络的控制器的主频
        ########

        ## 初始化车辆
        self.vehicleLocation = vehicleLocation
        self.vehicleCPU = vehicleCPU
        self.vehicleTaskprofit = vehicleTaskprofit
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
    def setCCPU(self,cCPU):
        self.cCpu = cCPU
    def reset(self):
        self.idletime = 0
        self.index = 0
     ########################################################################
    #########################  --计算效用-- ##################################

    ## params:action 保存卸载量
    ## output:utility1 价钱效用
    ##        utility2 实际效用
    def utility(self,action):
        #### ##
        # 理论计算        
        # for index in range(self.numofVehicle):
        #     taskCapa += action[index] * self.vehicle.taskprofit[1]
        # t_sum = taskCapa/self.CapacityofCPU # 总处理时延
        # _t_com = t_sum/self.numofVehicle ## 平均处理时延
        

        # 定义一个处理队列，保存正在运行的程序以及正在等待的程序
        # arrivingInterval = self.lamda
        computeQueue = []  ## 储存了每个任务的等待时间
        idletime = self.idletime
        for index in range(self.numofVehicle):
            taskload = action[index] * self.vehicle[index].taskprofit[1]
            arriveTime = self.index+self.sublamda[index]
            # taskloadSum += taskload
            if(idletime <= arriveTime): ## 说明是空闲的
                idletime = taskload/self.CapacityofCPU + arriveTime ## 更新计算完这个任务的时间
                computeQueue.append(taskload/self.CapacityofCPU)  ## 空闲的等待时间直接算就行了
                
            else:
                idletime = taskload/self.CapacityofCPU + idletime       ## 不是空闲的，就在当前idle的基础上累积，也是计算完这个任务的时间
                computeQueue.append(idletime - arriveTime)  ## 然后减去任务过来的时间

        ## 更新系统的idletime_last和index_last
        self.idletime_last = idletime


        utility1 = [] 
        utility2 = []
        for index in range(self.numofVehicle):
            localtime = self.vehicle[index].taskprofit[1]*(self.vehicle[index].taskprofit[0] - action[index])/self.vehicle[index].CapacityofCPU
            mectime = computeQueue[index]+(action[index]/self.R_v_v_(self.subBand,self.distance(self.vehicle[index].location,self.location)))
            computeCost = self.cCpu * computeQueue[index]
            utility1.append(self.vehicle[index].taskprofit[2] - max(localtime,mectime) - computeCost)
            utility2.append(self.vehicle[index].taskprofit[2] - max(localtime,mectime))
        return utility1,utility2
        
     ########################################################################

