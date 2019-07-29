# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

"""
该案例展示了一个离散决策变量的最小化目标的双目标优化问题。
min f1 = -25 * (x1 - 2)**2 - (x2 - 2)**2 - (x3 - 1)**2 - (x4 - 4)**2 - (x5 - 1)**2
min f2 = (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2 + (x4 - 1)**2 + (x5 - 1)**2
s.t.
x1 + x2 >= 2
x1 + x2 <= 6
x1 - x2 >= -2
x1 - 3*x2 <= 2
4 - (x3 - 3)**2 - x4 >= 0
(x5 - 3)**2 + x4 - 4 >= 0
x1,x2,x3,x4,x5 ∈ {0,1,2,3,4,5,6,7,8,9,10}
"""
a = 0.21
b = 0.50
c = 0.84
d = 0.93
j = 100
class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self, M = 2):
        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        Dim = 77 # 初始化Dim（决策变量维数）
        maxormins = [1] * M # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim # 决策变量下界
        ub = [1] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界
        ubin = [1] * Dim # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop): # 目标函数
        Vars = pop.Phen # 得到决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]
        x5 = Vars[:, [4]]
        x6 = Vars[:, [5]]
        x7 = Vars[:, [6]]
        x8 = Vars[:, [7]]
        x9 = Vars[:, [8]]
        x10 = Vars[:, [9]]
        x11 = Vars[:, [10]]
        x12 = Vars[:, [11]]
        x13 = Vars[:, [12]]
        x14 = Vars[:, [13]]
        x15 = Vars[:, [14]]
        x16 = Vars[:, [15]]
        x17 = Vars[:, [16]]
        x18 = Vars[:, [17]]
        x19 = Vars[:, [18]]
        x20 = Vars[:, [19]]
        x21 = Vars[:, [20]]
        x22 = Vars[:, [21]]
        x23 = Vars[:, [22]]
        x24 = Vars[:, [23]]
        x25 = Vars[:, [24]]
        x26 = Vars[:, [25]]
        x27 = Vars[:, [26]]
        x28 = Vars[:, [27]]
        x29 = Vars[:, [28]]
        x30 = Vars[:, [29]]
        x31 = Vars[:, [30]]
        x32 = Vars[:, [31]]
        x33 = Vars[:, [32]]
        x34 = Vars[:, [33]]
        x35 = Vars[:, [34]]
        x36 = Vars[:, [35]]
        x37 = Vars[:, [36]]
        x38 = Vars[:, [37]]
        x39 = Vars[:, [38]]
        x40 = Vars[:, [39]]
        x41 = Vars[:, [40]]
        x42 = Vars[:, [41]]
        x43 = Vars[:, [42]]
        x44 = Vars[:, [43]]
        x45 = Vars[:, [44]]
        x46 = Vars[:, [45]]
        x47 = Vars[:, [46]]
        x48 = Vars[:, [47]]
        x49 = Vars[:, [48]]
        x50 = Vars[:, [49]]
        x51 = Vars[:, [50]]
        x52 = Vars[:, [51]]
        x53 = Vars[:, [52]]
        x54 = Vars[:, [53]]
        x55 = Vars[:, [54]]
        x56 = Vars[:, [55]]
        x57 = Vars[:, [56]]
        x58 = Vars[:, [57]]
        x59 = Vars[:, [58]]
        x60 = Vars[:, [59]]
        x61 = Vars[:, [60]]
        x62 = Vars[:, [61]]
        x63 = Vars[:, [62]]
        x64 = Vars[:, [63]]
        x65 = Vars[:, [64]]
        x66 = Vars[:, [65]]
        x67 = Vars[:, [66]]
        x68 = Vars[:, [67]]
        x69 = Vars[:, [68]]
        x70 = Vars[:, [69]]
        x71 = Vars[:, [70]]
        x72 = Vars[:, [71]]
        x73 = Vars[:, [72]]
        x74 = Vars[:, [73]]
        x75 = Vars[:, [74]]
        x76 = Vars[:, [75]]
        x77 = Vars[:, [76]]
        
        f1 = c*1.5*x1+d*1.5*x2+d*1.5*x3+c*1.5*x4+a*1.5*x5+a*1.5*x6+a*1.5*x7+c*0.8*x8+\
             d*0.8*x9+d*0.8*x10+b*0.4*x11+b*0.4*x12+c*0.4*x13+c*0.4*x14+c*0.4*x15+b*0.4*x16+b*0.4*x17+\
             b*0.4*x18+b*0.4*x19+c*0.4*x20+d*0.4*x21+d*0.4*x22+c*0.4*x23+a*0.4*x24+ \
             c*2.4*x25+d*2.4*x26+d*2.4*x27+c*2.4*x28+a*2.4*x29+a*2.4*x30+a*2.4*x31+\
             a*2.4*x32+a*2.4*x33+a*2.4*x34+a*2.4*x35+a*2.4*x36+a*2.4*x37+\
             a*0.8*x38+a*0.8*x39+b*0.8*x40+b*0.8*x41+\
             a*1.3*x42+a*1.3*x43+b*1.3*x44+b*1.3*x45+\
             b*1.3*x46+c*1.3*x47+c*1.3*x48+c*1.3*x49+\
             b*1.3*x50+c*1.3*x51+d*1.3*x52+d*1.3*x53+  \
             a*0.35*x54+a*0.35*x55+b*0.35*x56+b*0.35*x57+ \
             b*0.35*x58+c*0.35*x59+c*0.35*x60+c*0.35*x61+\
             b*0.35*x62+c*0.35*x63+d*0.35*x64+d*0.35*x65+   \
             b*2*x66+b*2*x67+c*2*x68+c*2*x69+c*2*x70+b*2*x71+\
             b*2*x72+c*2*x73+d*2*x74+d*2*x75+c*2*x76+a*2*x77+   \
             0.3*9*j*( x1+x2+x3+x4+x5+x6+x7 -4)*( x1+x2+x3+x4+x5+x6+x7 -4)+\
             0.3*9*j*( x8+x9+x10-1 )*( x8+x9+x10-1 )+\
             0.3*9*j*( x11+x12+x13+x14+x15+x16+x17+x18+x19+x20+x21+x22+x23+x24 - 2)*( x11+x12+x13+x14+x15+x16+x17+x18+x19+x20+x21+x22+x23+x24 - 2)+\
             0.2*9*j*( x25+x26+x27+x28+x29+x30+x31+x32+x33+x34+x35+x36+x37 - 4)*( x25+x26+x27+x28+x29+x30+x31+x32+x33+x34+x35+x36+x37 - 4)+\
             0.2*9*j*( x38+x39+x40+x41 - 1)*( x38+x39+x40+x41 - 1)+\
            0.2*9*j*( x42+x43+x44+x45 - 1)*( x42+x43+x44+x45 - 1)+\
            0.3*9*j*( x46+x47+x48+x49 - 1)*( x46+x47+x48+x49 - 1)+\
            0.3*9*j*( x50+x51+x52+x53 - 2)*( x50+x51+x52+x53 - 2)+\
            0.3*9*j*( x54+x55+x56+x57 - 1)*( x54+x55+x56+x57 - 1)+\
            0.3*9*j*( x58+x59+x60+x61 - 1)*( x58+x59+x60+x61 - 1)+\
            0.3*9*j*( x62+x63+x64+x65 - 2)*( x62+x63+x64+x65 - 2)+\
            0.3*9*j*( x66+x67+x68+x69+x70+x71 - 3)*( x66+x67+x68+x69+x70+x71 - 3)+\
            1*9*j*( x72+x73+x74+x75+x76+x77 - 3)*( x72+x73+x74+x75+x76+x77 - 3);
        f2 = ( 1.5*( abs(x1-1) + abs(x2-1) + abs(x3-1) + abs(x4-1) + abs(x5) + abs(x6) + abs(x7))+\
                0.8*( abs(x8-1) + abs(x9) + abs(x10) )+\
    0.4*( abs(x11) + abs(x12) + abs(x13) + abs(x14) + abs(x15) + abs(x16) + abs(x17) +\
     abs(x18) + abs(x19) + abs(x20) + abs(x21) + abs(x22) + abs(x23-1) + abs(x24-1) )+\
     2.4*( abs(x25-1) + abs(x26-1) + abs(x27-1) + abs(x28-1) + abs(x29) + abs(x30) + abs(x31) +\
      abs(x32) + abs(x33) + abs(x34) + abs(x35) + abs(x36) + abs(x37) )+\
      0.8*( abs(x38) + abs(x39) + abs(x40-1) + abs(x41)) +\
      1.3*( abs(x42) + abs(x43) + abs(x44-1) + abs(x45) +  abs(x46) + abs(x47-1) + abs(x48) + abs(x49) +\
      abs(x50) + abs(x51-1) + abs(x52-1) + abs(x53)  ) +\
      0.35*( abs(x54) + abs(x55) + abs(x56-1) + abs(x57) +abs(x58) +\
      abs(x59-1) + abs(x60) + abs(x61) + abs(x62) + abs(x63-1) + abs(x64-1) + abs(x65) ) +\
      2 * ( abs(x66) + abs(x67) + abs(x68-1) + abs(x69-1) + abs(x70-1) + abs(x71)  +\
      abs(x72) + abs(x73-1) + abs(x74-1) + abs(x75-1) + abs(x76) + abs(x77) )      )/37.25 - 1;
#        # 利用罚函数法处理约束条件
#        idx1 = np.where(x1 + x2 < 2)[0]
#        idx2 = np.where(x1 + x2 > 6)[0]
#        idx3 = np.where(x1 - x2 < -2)[0]
#        idx4 = np.where(x1 - 3*x2 > 2)[0]
#        idx5 = np.where(4 - (x3 - 3)**2 - x4 < 0)[0]
#        idx6 = np.where((x5 - 3)**2 + x4 - 4 < 0)[0]
#        exIdx = np.unique(np.hstack([idx1, idx2, idx3, idx4, idx5, idx6])) # 得到非可行解的下标
#        f1[exIdx] = f1[exIdx] + np.max(f1) - np.min(f1)
#        f2[exIdx] = f2[exIdx] + np.max(f2) - np.min(f2)
        # 利用可行性法则处理约束条件
##        pop.CV = np.hstack([2 - x1 - x2,
##                            x1 + x2 - 6,
##                            -2 - x1 + x2,
##                            x1 - 3*x2 - 2,
##                            (x3 - 3)**2 + x4 - 4,
##                            4 - (x5 - 3)**2 - x4])
        pop.ObjV = np.hstack([f1, f2]) # 把求得的目标函数值赋值给种群pop的ObjV
    
