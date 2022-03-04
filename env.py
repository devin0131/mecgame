from asyncio import Task, tasks
from dis import dis
from distutils.command.config import LANG_EXT
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
    def __init__(self,numofVehicle,capacityofCPU,location,vehicleLocation,vehicleCPU,vehicleTaskprofit,lamda = 2) -> None:

        ## 系统参量
        self.numofVehicle = numofVehicle
        self.lamda = lamda

        ## 信道参量
        self.L0 = 61094
        self.P_tx_v = 1
        self.P_w_r = pNoise(26,10e6)
        self.subBand = 10e6
        self.location = location
        self.alpha = 2.75
        #######

        # ## 代价参量
        # self.cSpec = np.zeros(self.numofVehicle)
        # self.cCpu = np.zeros(self.numofVehicle)
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

    #########################  --计算效用-- ##################################

    ###################
    ## params:action 保存卸载量
    def utility(self,action):
        #### ##
        # 理论计算        
        # for index in range(self.numofVehicle):
        #     taskCapa += action[index] * self.vehicle.taskprofit[1]
        # t_sum = taskCapa/self.CapacityofCPU # 总处理时延
        # _t_com = t_sum/self.numofVehicle ## 平均处理时延
        

        # 定义一个处理队列，保存正在运行的程序以及正在等待的程序
        arrivingInterval = 1/self.lamda
        taskqueue = []
        computeQueue = []  ## 储存了每个任务的等待时间
        # taskloadSum = 0
        idletime = 0
        for index in range(self.numofVehicle):
            taskload = action[index] * self.vehicle[index].taskprofit[1]
            taskqueue.append(taskload)
            # taskloadSum += taskload

            if(idletime <= index*arrivingInterval): ## 说明是空闲的
                idletime = taskload/self.CapacityofCPU + index*arrivingInterval
                computeQueue.append(idletime - index*arrivingInterval)
            else:
                idletime = idletime + taskload/self.CapacityofCPU
                computeQueue.append(idletime - index*arrivingInterval)
        utility = []
        for index in range(self.numofVehicle):
            localtime = self.vehicle[index].taskprofit[1]*(self.vehicle[index].taskprofit[0] - action[index])/self.vehicle[index].CapacityofCPU
            distance = self.distance(self.vehicle[index].location,self.location)
            mectime = computeQueue[index]+(action[index]/self.R_v_v_(self.subBand,distance))
            utility.append(self.vehicle[index].taskprofit[2] - max(localtime,mectime))
        return utility
        

        
        
        



     ########################################################################

