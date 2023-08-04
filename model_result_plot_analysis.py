# -*- coding: utf-8 -*-
# @Time    : 2021/11/27 16:28
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : model_result_plot_analysis.py
# @Software: PyCharm
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode
import sys

# cf.go_offline()
# init_notebook_mode
# sys.path.insert(0, '.')
from TrainAndEvaluate.plotting import Plotter
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:95% !important; }</style>"))

# res_path = './MultiWaveExperiment/MultiWaveCaseStudy/20220429/'
res_path = './cache/uncategorized'

plotter = Plotter(res_path, model_id='-1')

plotter.plot_analysis(
    feature=22,
    plot_train=False,
    plot_errors=True,
    plot_feature_anom=True,
    output_filename="./e-2.html"
)

print("Analysis Completed!")
