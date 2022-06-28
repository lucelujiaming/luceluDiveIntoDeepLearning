import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
 
#        plt.figure(1)
#        t = [0]
#        t_now = 0
#        m = [sin(t_now)]
#         
#        plt.ion() #开启interactive mode 成功的关键函数
#        plt.show()
#        for i in range(200):
#            t_now = i*0.1
#            t.append(t_now)#模拟数据增量流入
#            m.append(sin(t_now))#模拟数据增量流入
#            plt.plot(t,m,'-r')
#            plt.draw()#注意此函数需要调用
#            time.sleep(0.01)
#        plt.ioff()
#        plt.show()

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__' :
    t1 = np.arange(0, 30, 0.1)
    plt.figure()
    plt.ion()
    for i in range(100):
        plt.ylim(-10, 10) #此处限制了一下y轴坐标最大最小值，防止刻度变化，不利于观察。
        plt.plot(t1, 0.1*i*np.sin(t1 + 0.1 * i))
        plt.plot(t1, 0.1 * (i -100) * np.cos(t1 + 0.2 * i))
        plt.pause(0.01)
        plt.clf()
    plt.ioff()
    plt.show()
