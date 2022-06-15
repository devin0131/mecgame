import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.font_manager as fm
## rcParams['font.family'] = 'STZhongsong,Times News Roman'
font = fm.FontProperties(fname='/usr/share/fonts/custom/KaiTi_GB2312.ttf')



with open("./varCo.pkl","rb") as f:
    varCo = pickle.load(f)
with open("./varNoCo.pkl","rb") as f:
    varNoCo = pickle.load(f)
iteratorTime = 10
numofVehicle = 6
line_shape = ["*-","^-",".-","s-","D-","1-"]
legend = ["合作博弈","非合作博弈","车辆3","车辆4","车辆5","车辆6"]
x = range(1,iteratorTime+1)
varCo = varCo[:10]
varNoCo = varNoCo[:10]
plt.figure()
plt.plot(x,np.array(varCo),line_shape[0],label=legend[0],linewidth=1.8,markersize=8)
plt.plot(x,np.array(varNoCo),line_shape[1],label=legend[1],linewidth=1.8,markersize=8)
fontsize = 9
plt.grid()
plt.legend(fontsize=fontsize)
plt.xticks(x,fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim(min(x),max(x))
plt.hlines(0, 1, 20,color="black",linestyles="dashdot",linewidths=3)#横线
# plt.ylim(0.2,1)
plt.xlabel("迭代次数",fontsize=fontsize + 3,fontproperties=font)
plt.ylabel("效用方差",fontsize = fontsize+3,fontproperties=font)
plt.savefig("var.pdf")
plt.plot()
