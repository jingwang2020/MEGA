# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 14:38
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : paper_figure.py
# @Software: PyCharm

from plotly.graph_objs import *
import plotly.offline
from plotly import subplots
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


## 参数敏感性分析实验
def paramFigureA():

    x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    MSL_a_y = [0.9037, 0.9216, 0.8873, 0.8943, 0.8895, 0.9165, 0.8810, 0.8754]
    SMAP_a_y = [0.8091, 0.8028, 0.8263, 0.7985, 0.8213, 0.8326, 0.8018, 0.7905]
    SMD_a_y = [0.9413, 0.9611, 0.9500, 0.9310, 0.9621, 0.9710, 0.9341, 0.9354]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_a_y,'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_a_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_a_y, 's:', color='red', linewidth=2.7)
        # plt.scatter(x, MSL_a_y, marker='+', s=3.0, c='blue')
        # plt.legend(["MSL", "SMAP", "SMD"], fontsize=20)


        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 3.6)
        plt.ylim(0.75, 1.0)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        # 刻度间隔
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        # 坐标轴粗细
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramA.jpg")
        plt.show()


def paramFigureB():
    x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    MSL_b_y = [0.8734, 0.8821, 0.8934, 0.9088, 0.9106, 0.9015, 0.8836, 0.9099]
    SMAP_b_y = [0.7823, 0.8034, 0.8233, 0.8179, 0.8301, 0.7989, 0.7834, 0.8059]
    SMD_b_y = [0.9231, 0.9411, 0.9341, 0.9103, 0.9425, 0.9316, 0.9133, 0.9354]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_b_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_b_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_b_y, 's:', color='red', linewidth=2.7)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 3.6)
        plt.ylim(0.75, 0.95)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramB.jpg")
        plt.show()


def paramFigureC():
    x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    MSL_c_y = [0.8738, 0.8857, 0.9032, 0.8613, 0.8827, 0.8796, 0.8702, 0.8629]
    SMAP_c_y = [0.8301, 0.8112, 0.7916, 0.8323, 0.8046, 0.8233, 0.7834, 0.8023]
    SMD_c_y = [0.9118, 0.8778, 0.8819, 0.8689, 0.8898, 0.9156, 0.9061, 0.9261]


    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_c_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_c_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_c_y, 's:', color='red', linewidth=2.7)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 3.6)
        plt.ylim(0.75, 0.95)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramC.jpg")
        plt.show()


def paramFigureD():
    x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    MSL_d_y = [0.8774, 0.8916, 0.8701, 0.9119, 0.8916, 0.8893, 0.8636, 0.8756]
    SMAP_d_y = [0.8164, 0.7946, 0.8215, 0.7823, 0.7936, 0.8116, 0.7896, 0.7962]
    SMD_d_y = [0.9065, 0.9141, 0.8971, 0.9310, 0.9441, 0.9556, 0.9241, 0.9354]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_d_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_d_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_d_y, 's:', color='red', linewidth=2.7)
        plt.legend(["MSL", "SMAP", "SMD"], fontsize=17)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 3.6)
        plt.ylim(0.75, 1.0)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramD.jpg")
        plt.show()


def paramFigureE():
    x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    MSL_e_y = [0.8850, 0.9198, 0.8796, 0.9065, 0.8664, 0.9134, 0.8986, 0.9146]
    SMAP_e_y = [0.8146, 0.8046, 0.8212, 0.8065, 0.8116, 0.8098, 0.7995, 0.7754]
    SMD_e_y = [0.9265, 0.9389, 0.9515, 0.9089, 0.9185, 0.9464, 0.9613, 0.9364]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_e_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_e_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_e_y, 's:', color='red', linewidth=2.7)
        # plt.legend(["MSL", "SMAP", "SMD"], fontsize=20)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 3.6)
        plt.ylim(0.75, 1.0)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramE.jpg")
        plt.show()


## 参数敏感性分析实验2: 0.0~1.0区间
def paramFigureADetail():

    x = [0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9, 1.0] # 0.1, 0.5, 1.0
    MSL_a_y = [0.9037,0.8973, 0.8900,0.9019, 0.9216,0.9289,0.9100,0.8910,0.8899, 0.8873]
    SMAP_a_y = [0.8091,0.8053,0.8041, 0.7956,0.8028,0.8099,0.8300,0.8250,0.8291, 0.8263]
    SMD_a_y = [0.9413,0.9316,0.9516,0.9601, 0.9611,0.9689,0.9579,0.9516,0.9401, 0.9500]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_a_y,'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_a_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_a_y, 's:', color='red', linewidth=2.7)
        # plt.scatter(x, MSL_a_y, marker='+', s=3.0, c='blue')
        plt.legend(["MSL", "SMAP", "SMD"], fontsize=15)


        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 1.1)
        plt.ylim(0.75, 1.0)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        # 刻度间隔
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        # 坐标轴粗细
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramA_detail.jpg")
        plt.show()


def paramFigureBDetail():
    x = [0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9, 1.0]
    MSL_b_y = [0.8734,0.8619,0.8799,0.8801, 0.8821,0.8999,0.8910,0.8921,0.8930, 0.8934]
    SMAP_b_y = [0.7823,0.7910,0.8120,0.8011, 0.8034,0.8102,0.80,0.8189,0.8219, 0.8233]
    SMD_b_y = [0.9231,0.9179,0.9310,0.9399, 0.9411,0.9589,0.9400,0.9379,0.9350, 0.9341]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_b_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_b_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_b_y, 's:', color='red', linewidth=2.7)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 1.1)
        plt.ylim(0.75, 0.98)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramB_detail.jpg")
        plt.show()


def paramFigureCDetail():
    x = [0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9, 1.0]
    MSL_c_y = [0.8738, 0.8755,0.8899,0.8831, 0.8857,0.8799,0.9016,0.9156,0.9089, 0.9032]
    SMAP_c_y = [0.8301,0.8200,0.8316,0.8101, 0.8112,0.8179,0.8010,0.7811,0.7920, 0.7916]
    SMD_c_y = [0.9118,0.9023,0.9099,0.8920, 0.8778,0.8899,0.8923,0.8910,0.8850, 0.8819]


    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_c_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_c_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_c_y, 's:', color='red', linewidth=2.7)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 1.1)
        plt.ylim(0.75, 0.95)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramC_detail.jpg")
        plt.show()


def paramFigureDDetail():
    x = [0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9, 1.0]
    MSL_d_y = [0.8774,0.8716,0.8699,0.8886, 0.8916,0.9000,0.8900,0.8810,0.8789, 0.8701]
    SMAP_d_y = [0.8164,0.8100,0.7989,0.7950, 0.7946,0.7816,0.7889,0.7953,0.8154, 0.8215]
    SMD_d_y = [0.8865,0.9088,0.9299,0.9313, 0.9141,0.9100,0.8995,0.8989,0.8980, 0.8971]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_d_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_d_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_d_y, 's:', color='red', linewidth=2.7)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 1.1)
        plt.ylim(0.75, 0.95)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramD_detail.jpg")
        plt.show()


def paramFigureEDetail():
    x = [0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9, 1.0]
    MSL_e_y = [0.8850,0.8899,0.9056,0.9120, 0.9198,0.92,0.9016,0.8950,0.8813, 0.8796]
    SMAP_e_y = [0.8146,0.8250,0.8100,0.8079, 0.8046,0.7999,0.8150,0.8189,0.8200, 0.8212]
    SMD_e_y = [0.9265,0.9270,0.9300,0.9436, 0.9389,0.9370,0.9500,0.9510,0.9677, 0.9515]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_e_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_e_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_e_y, 's:', color='red', linewidth=2.7)
        # plt.legend(["MSL", "SMAP", "SMD"], fontsize=20)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 1.1)
        plt.ylim(0.75, 1.0)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramE_detail.jpg")
        plt.show()


def paramFigureOverallDetailA():

    x = [0.1,    0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,    1.0,   1.1,   1.2,   1.3,   1.4,    1.5,   1.6,   1.7,   1.8,   1.9,    2.0,   2.1,   2.2,   2.3,   2.4,    2.5,   2.6,   2.7,   2.8,   2.9,    3.0,   3.1,   3.2,   3.3,   3.4,    3.5]  # 0.1, 0.5, 1.0
    MSL_a_y = [0.9037, 0.8973, 0.8900, 0.9019, 0.9216, 0.9289, 0.9100, 0.8910, 0.8899, 0.8873,0.8792,0.8790,0.8890,0.8981, 0.8943,0.8940,0.8962,0.8900,0.8823, 0.8895,0.8932,0.8810,0.8999,0.9091, 0.9165,0.8978,0.9120,0.9100,0.8956, 0.8810,0.8756,0.8923,0.8956,0.8812, 0.8754]
    SMAP_a_y = [0.8091, 0.8053, 0.8041, 0.7956, 0.8028, 0.8099, 0.8300, 0.8250, 0.8291,0.8263,0.8199,0.8100,0.8200,0.8121, 0.7985,0.7970,0.8023,0.7991,0.8102, 0.8213,0.7916,0.8012,0.8103,0.8219, 0.8326,0.8199,0.8200,0.8156,0.8065, 0.8018,0.8256,0.8219,0.8106,0.8020, 0.7905]
    SMD_a_y = [0.9413, 0.9316, 0.9516, 0.9601, 0.9611, 0.9689, 0.9579, 0.9516, 0.9401, 0.9500,0.9519,0.9423,0.9156,0.9199, 0.9310,0.9199,0.9230,0.9300,0.9502, 0.9621,0.9519,0.9556,0.9319,0.9529, 0.9710,0.9569,0.9610,0.9323,0.9102, 0.9341,0.9102,0.9156,0.9299,0.9420, 0.9354]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_a_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_a_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_a_y, 's:', color='red', linewidth=2.7)
        font = {
                'size': 20,
                }
        plt.ylabel("F1 score",fontdict=font)
        plt.legend(["MSL", "SMAP", "SMD"], fontsize=10)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 3.6)
        plt.ylim(0.75, 1.0)


        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        # 刻度间隔
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        # 坐标轴粗细
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramA_overall_detail.jpg")
        plt.show()


def paramFigureOverallDetailB():

    x = [0.1,    0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,    1.0,   1.1,   1.2,   1.3,   1.4,    1.5,   1.6,   1.7,   1.8,   1.9,    2.0,   2.1,   2.2,   2.3,   2.4,    2.5,   2.6,   2.7,   2.8,   2.9,    3.0,   3.1,   3.2,   3.3,   3.4,    3.5]  # 0.1, 0.5, 1.0
    MSL_b_y = [0.8734, 0.8619, 0.8799, 0.8801, 0.8821, 0.8999, 0.8910, 0.8921, 0.8930, 0.8934,0.9021,0.9100,0.8906,0.9044, 0.9088,0.9100,0.8966,0.8899,0.8960, 0.9106,0.9156,0.9158,0.9099,0.9000, 0.9015,0.8961,0.8956,0.8912,0.8856, 0.8836,0.8879,0.8891,0.9034,0.8989, 0.9099]
    SMAP_b_y = [0.7823, 0.7910, 0.8120, 0.8011, 0.8034, 0.8102, 0.80, 0.8189, 0.8219, 0.8233, 0.8256,0.8271,0.8200,0.8189, 0.8179,0.8106,0.8089,0.8156,0.8279, 0.8301,0.8189,0.8269,0.8100,0.7956, 0.7989,0.8000,0.8206,0.8101,0.8016, 0.7834,0.7918,0.8023,0.8106,0.8133, 0.8059]
    SMD_b_y = [0.9231, 0.9179, 0.9310, 0.9399, 0.9411, 0.9589, 0.9400, 0.9379, 0.9350, 0.9341,0.9368,0.9216,0.9256,0.9178, 0.9103,0.9100,0.8988,0.9156,0.9416, 0.9425,0.9400,0.9368,0.9216,0.9300, 0.9316,0.9342,0.9328,0.9281,0.9201, 0.9133,0.9189,0.9200,0.9106,0.9256, 0.9354]
    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_b_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_b_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_b_y, 's:', color='red', linewidth=2.7)
        plt.legend(["MSL", "SMAP", "SMD"], fontsize=10)

        font = {
            'size': 20,
        }
        plt.ylabel("F1 score", fontdict=font)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 3.6)
        plt.ylim(0.75, 1.0)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        # 刻度间隔
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        # 坐标轴粗细
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramB_overall_detail.jpg")
        plt.show()


def paramFigureOverallDetailC():

    x = [0.1,    0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,    1.0,   1.1,   1.2,   1.3,   1.4,    1.5,   1.6,   1.7,   1.8,   1.9,    2.0,   2.1,   2.2,   2.3,   2.4,    2.5,   2.6,   2.7,   2.8,   2.9,    3.0,   3.1,   3.2,   3.3,   3.4,    3.5]  # 0.1, 0.5, 1.0
    MSL_c_y = [0.8738, 0.8755, 0.8899, 0.8831, 0.8857, 0.8799, 0.9016, 0.9156, 0.9089, 0.9032,0.9056,0.8913,0.8812,0.8900, 0.8613,0.8789,0.8723,0.8799,0.8810, 0.8827,0.8936,0.8989,0.8756,0.8788, 0.8796,0.8712,0.8700,0.8589,0.8612, 0.8702,0.8723,0.8732,0.8765,0.8619, 0.8629]
    SMAP_c_y = [0.8301, 0.8200, 0.8316, 0.8101, 0.8112, 0.8179, 0.8010, 0.7811, 0.7920,0.7916,0.7979,0.8000,0.7916,0.8123, 0.8323,0.8356,0.8300,0.8216,0.8109, 0.8046,0.8189,0.8100,0.8192,0.8200, 0.8233,0.8200,0.8286,0.8100,0.8012, 0.7834,0.7912,0.7889,0.8012,0.8103, 0.8023]
    SMD_c_y = [0.9118, 0.9023, 0.9099, 0.8920, 0.8778, 0.8899, 0.8923, 0.8910, 0.8850, 0.8819,0.8899,0.8712,0.8787,0.8700, 0.8689,0.8710,0.8632,0.8789,0.8810, 0.8898,0.8916,0.8900,0.9018,0.9023, 0.9156,0.9199,0.9100,0.8956,0.8999, 0.9061,0.9012,0.9095,0.8812,0.9032, 0.9261]
    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_c_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_c_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_c_y, 's:', color='red', linewidth=2.7)
        plt.legend(["MSL", "SMAP", "SMD"], fontsize=10)
        font = {
            'size': 20,
        }
        plt.ylabel("F1 score", fontdict=font)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 3.6)
        plt.ylim(0.75, 1.0)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        # 刻度间隔
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        # 坐标轴粗细
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramC_overall_detail.jpg")
        plt.show()


def paramFigureOverallDetailD():

    x = [0.1,    0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,    1.0,   1.1,   1.2,   1.3,   1.4,    1.5,   1.6,   1.7,   1.8,   1.9,    2.0,   2.1,   2.2,   2.3,   2.4,    2.5,   2.6,   2.7,   2.8,   2.9,    3.0,   3.1,   3.2,   3.3,   3.4,    3.5]  # 0.1, 0.5, 1.0
    MSL_d_y = [0.8774, 0.8716, 0.8699, 0.8886, 0.8916, 0.9000, 0.8900, 0.8810, 0.8789, 0.8701,0.8799,0.8756,0.8912,0.9099, 0.9119,0.9156,0.9102,0.9187,0.9045, 0.8916,0.8910,0.8891,0.8712,0.8834, 0.8893,0.8810,0.8923,0.8912,0.8832, 0.8636,0.8689,0.8819,0.8779,0.8770, 0.8756]
    SMAP_d_y = [0.8164, 0.8100, 0.7989, 0.7950, 0.7946, 0.7816, 0.7889, 0.7953, 0.8154,0.8215,0.8223,0.8279,0.8100,0.8079, 0.7823,0.8012,0.8079,0.8000,0.7979, 0.7936,0.7846,0.7899,0.8013,0.8080, 0.8116,0.8219,0.8108,0.8102,0.7979, 0.7896,0.7916,0.7878,0.8015,0.8089, 0.7962]
    SMD_d_y = [0.8865, 0.9088, 0.9299, 0.9313, 0.9141, 0.9100, 0.8995, 0.8989, 0.8980, 0.8971,0.8991,0.8816,0.9012,0.9345, 0.9310,0.9456,0.9102,0.9156,0.9265, 0.9441,0.9440,0.9400,0.9312,0.9565, 0.9556,0.9500,0.9423,0.9410,0.9489, 0.9241,0.8912,0.8900,0.9102,0.9254, 0.9354]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_d_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_d_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_d_y, 's:', color='red', linewidth=2.7)
        plt.legend(["MSL", "SMAP", "SMD"], fontsize=10)
        font = {
            'size': 20,
        }
        plt.ylabel("F1 score", fontdict=font)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 3.6)
        plt.ylim(0.75, 1.0)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        # 刻度间隔
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        # 坐标轴粗细
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramD_overall_detail.jpg")
        plt.show()


def paramFigureOverallDetailE():

    x = [0.1,    0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8,    0.9,    1.0,   1.1,   1.2,   1.3,   1.4,    1.5,   1.6,   1.7,   1.8,   1.9,    2.0,   2.1,   2.2,   2.3,   2.4,    2.5,   2.6,   2.7,   2.8,   2.9,    3.0,   3.1,   3.2,   3.3,   3.4,    3.5]  # 0.1, 0.5, 1.0
    MSL_e_y = [0.8850, 0.8899, 0.9056, 0.9120, 0.9198, 0.92,   0.9016, 0.8950, 0.8813, 0.8796,0.8701,0.8679,0.8819,0.8810, 0.9065,0.9156,0.9150,0.9078,0.9100, 0.8664,0.8723,0.8691,0.8812,0.8918, 0.9134,0.9019,0.9012,0.9156,0.8999, 0.8986,0.8980,0.9069,0.9216,0.9200, 0.9146]
    SMAP_e_y = [0.8146, 0.8250, 0.8100, 0.8079, 0.8046, 0.7999, 0.8150, 0.8189, 0.8200,0.8212,0.8289,0.8300,0.8012,0.8156, 0.8065,0.8054,0.8000,0.8165,0.8189, 0.8116,0.7912,0.8000,0.8106,0.8078, 0.8098,0.8178,0.8100,0.8156,0.7865, 0.7995,0.8021,0.8099,0.8165,0.8000, 0.7754]
    SMD_e_y = [0.9265, 0.9270, 0.9300, 0.9436, 0.9389, 0.9370, 0.9500, 0.9510, 0.9677, 0.9515,0.9465,0.9578,0.9302,0.9018, 0.9089,0.8978,0.9173,0.9384,0.9263, 0.9185,0.9019,0.9341,0.9300,0.9239, 0.9464,0.9446,0.9219,0.9636,0.9536, 0.9613,0.9632,0.9105,0.9100,0.9279, 0.9364]

    with plt.style.context(['science', 'no-latex']):
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        fig0 = plt.figure(figsize=(8, 4))
        plt.plot(x, MSL_e_y, 'o-', color='blue', linewidth=2.7)
        plt.plot(x, SMAP_e_y, 'v--', color='green', linewidth=2.7)
        plt.plot(x, SMD_e_y, 's:', color='red', linewidth=2.7)
        plt.legend(["MSL", "SMAP", "SMD"], fontsize=10)
        font = {
            'size': 20,
        }
        plt.ylabel("F1 score", fontdict=font)

        plt.yticks(fontproperties='Times New Roman', size=20)  # 设置大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.xlim(0.05, 3.6)
        plt.ylim(0.75, 1.0)

        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        # Scale interval
        y_major_locator = MultipleLocator(0.1)
        x_major_locator = MultipleLocator(0.5)
        TK = plt.gca()
        TK.yaxis.set_major_locator(y_major_locator)
        TK.xaxis.set_major_locator(x_major_locator)
        # Axis Thickness
        TK.spines['bottom'].set_linewidth(2)
        TK.spines['left'].set_linewidth(2)
        TK.spines['top'].set_linewidth(2)
        TK.spines['right'].set_linewidth(2)
        plt.savefig("./Save_Component_Jpg/paramFigure/paramE_overall_detail.jpg")
        plt.show()


if __name__ == '__main__':
    # paramFigureA()
    # paramFigureB()
    # paramFigureC()
    # paramFigureD()
    # paramFigureE()

    # paramFigureADetail()
    # paramFigureBDetail()
    # paramFigureCDetail()
    # paramFigureDDetail()
    # paramFigureEDetail()

    # paramFigureOverallDetailA()
    # paramFigureOverallDetailB()
    # paramFigureOverallDetailC()
    # paramFigureOverallDetailD()
    paramFigureOverallDetailE()
