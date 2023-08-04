# -*- coding: utf-8 -*-
# @Time    : 2021/11/27 16:33
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : plotting.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import plotly as py
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import cufflinks as cf
cf.go_offline()


class Plotter:

    """
    Class for visualizing results of anomaly detection.
    Includes visualization of forecasts, reconstructions, anomaly scores, predicted and actual anomalies
    Plotter-class inspired by TelemAnom (https://github.com/khundman/telemanom)
    """

    def __init__(self, result_path, model_id='-1'):
        self.result_path = result_path
        self.model_id = model_id
        self.train_output = None
        self.test_output = None
        self.labels_available = True
        self.pred_cols = None
        self._load_results()
        self.train_output["timestamp"] = self.train_output.index
        self.test_output["timestamp"] = self.test_output.index

        # Use the correct csv name of your dataset
        # config_path = f"{self.result_path}/MSL_statistics.csv"
        config_path = f"{self.result_path}/statistics.csv"

        confi_df = pd.read_csv(config_path)
        self.lookback = confi_df['window_size'][0]

        if "SMD" in self.result_path.upper():
            self.pred_cols = [f"feat_{i}" for i in range(get_data_dim("machine"))]
        elif "SMAP" in self.result_path.upper() or "MSL" in self.result_path.upper():
            self.pred_cols = ["feat_1"]

    def _load_results(self):

        print(f"Loading results of {self.result_path}")
        train_output = pd.read_pickle(f"{self.result_path}/train_output.pkl")
        test_output = pd.read_pickle(f"{self.result_path}/test_output.pkl")

        # Because for SMAP and MSL only one feature is predicted
        if 'SMAP' in self.result_path or 'MSL' in self.result_path:
            train_output[f'A_Pred_0'] = train_output['A_Pred_Global']
            train_output[f'A_Score_0'] = train_output['A_Score_Global']
            train_output[f'Thresh_0'] = train_output['Thresh_Global']

            test_output[f'A_Pred_0'] = test_output['A_Pred_Global']
            test_output[f'A_Score_0'] = test_output['A_Score_Global']
            test_output[f'Thresh_0'] = test_output['Thresh_Global']

        self.train_output = train_output
        self.test_output = test_output

    def create_shapes(self, ranges, sequence_type, _min, _max, plot_values, is_test=True, xref=None, yref=None):
        """
        Create shapes for regions to highlight in plotly (true and predicted anomaly sequences).

        :param ranges: tuple of start and end indices for anomaly sequences for a feature
        :param sequence_type: "predict" if predicted values else "true" if actual values. Determines colors.
        :param _min: min y value of series
        :param _max: max y value of series
        :param plot_values: dictionary of different series to be plotted

        :return: list of shapes specifications for plotly
        """

        if _max is None:
            _max = max(plot_values["scores"])

        if sequence_type is None:
            color = "blue"
        else:
            color = "red" if sequence_type == "true" else "blue"
        shapes = []

        for r in ranges:
            w = 5
            x0 = r[0] - w
            x1 = r[1] + w
            shape = {
                "type": "rect",
                "x0": x0,
                "y0": _min,
                "x1": x1,
                "y1": _max,
                "fillcolor": color,
                "opacity": 0.08,
                "line": {
                    "width": 0,
                },
                "name": "true_label"
            }
            if xref is not None:
                shape["xref"] = xref
                shape["yref"] = yref

            shapes.append(shape)

        return shapes

    @staticmethod
    def get_anomaly_sequences(values):
        splits = np.where(values[1:] != values[:-1])[0] + 1
        if values[0] == 1:
            splits = np.insert(splits, 0, 0)

        a_seqs = []
        for i in range(0, len(splits) - 1, 2):
            a_seqs.append([splits[i], splits[i + 1] - 1])

        if len(splits) % 2 == 1:
            a_seqs.append([splits[-1], len(values) - 1])

        return a_seqs

    def plot_feature(self, feature, plot_train=False, plot_errors=True, plot_feature_anom=False, start=None, end=None):
        # Plotting ground truth, reconstructed values; ground truth and predicted labels
        test_copy = self.test_output.copy()

        if start is not None and end is not None:
            assert start < end
        if start is not None:
            test_copy = test_copy.iloc[start:, :]
        if end is not None:
            start = 0 if start is None else start
            test_copy = test_copy.iloc[: end - start, :]

        plot_data = [test_copy]

        if plot_train:
            train_copy = self.train_output.copy()
            plot_data.append(train_copy)

        for nr, data_copy in enumerate(plot_data):
            is_test = nr == 0

            i = feature
            plot_values = {
                "timestamp": data_copy["timestamp"].values,
                "y_recon": data_copy[f"Recon_{i}"].values,
                "y_true": data_copy[f"True_{i}"].values,
            }

            anomaly_sequences = {
                # "pred": self.get_anomaly_sequences(data_copy[f"A_Pred_{i}"].values),
                "true": self.get_anomaly_sequences(data_copy["A_True_Global"].values),
            }

            if is_test and start is not None:
                # anomaly_sequences['pred'] = [[s+start, e+start] for [s, e] in anomaly_sequences['pred']]
                anomaly_sequences['true'] = [[s+start, e+start] for [s, e] in anomaly_sequences['true']]

            y_min = 1.1 * plot_values["y_true"].min()
            y_max = 1.1 * plot_values["y_true"].max()

            y_shapes = self.create_shapes(anomaly_sequences["true"], "true", y_min, y_max, plot_values, is_test=is_test)
            # e_shapes = self.create_shapes(anomaly_sequences["pred"], "predicted", 0, e_max, plot_values, is_test=is_test)
            # if self.labels_available and ('SMAP' in self.result_path.upper() or 'MSL' in self.result_path.upper()):
                # y_shapes += self.create_shapes(anomaly_sequences["true"], "true", y_min, y_max, plot_values, is_test=is_test)
                # e_shapes += self.create_shapes(anomaly_sequences["true"], "true", 0, e_max, plot_values, is_test=is_test)

            y_df = pd.DataFrame(
                {
                    "timestamp": plot_values["timestamp"].reshape(-1, ),
                    "y_recon": plot_values["y_recon"].reshape(-1, ),
                    "y_true": plot_values["y_true"].reshape(-1, )
                }
            )

            data_type = "Test data" if is_test else "Train data"
            y_layout = {
                "title": f"{data_type} | Reconstruction vs true value for {self.pred_cols[i] if self.pred_cols is not None else ''} ",
                "showlegend": True,
                # "height": 400,
                # "width": 1100,
            }

            if plot_feature_anom:
                y_layout["shapes"] = y_shapes

            lines = [
                go.Scatter(
                    x=y_df["timestamp"],
                    y=y_df["y_true"],
                    line_color="rgb(0, 204, 150, 0.5)",
                    name="y_true",
                    line=dict(width=1)),
                go.Scatter(
                    x=y_df["timestamp"],
                    y=y_df["y_recon"],
                    line_color="rgb(31, 119, 180, 1)",
                    name="y_recon",
                    line=dict(width=1)),
            ]

            fig = go.Figure(data=lines, layout=y_layout)
            py.offline.plot(fig, filename="./test.html")

    def plot_analysis(self, feature, plot_train=False, plot_errors=True, plot_feature_anom=False, start=None, end=None, output_filename=None):
        """
        :param feature: the feature to draw
        :param plot_train:
        :param plot_errors:
        :param plot_feature_anom:
        :param start: The starting point of the curve to be drawn
        :param end: The end point of the curve to be drawn
        :return: Draw two subplots, one containing the true anomaly labels and one containing the model-predicted labels
        """
        test_copy = self.test_output.copy()

        if start is not None and end is not None:
            assert start < end
        if start is not None:
            test_copy = test_copy.iloc[start:, :]
        if end is not None:
            start = 0 if start is None else start
            test_copy = test_copy.iloc[: end - start, :]

        plot_data = [test_copy]

        if plot_train:
            train_copy = self.train_output.copy()
            plot_data.append(train_copy)

        for nr, data_copy in enumerate(plot_data):
            is_test = nr == 0

            i = feature
            plot_values = {
                "timestamp": data_copy["timestamp"].values,
                "y_recon": data_copy[f"Recon_{i}"].values,
                "y_true": data_copy[f"True_{i}"].values,
            }

            anomaly_sequences = {
                "pred": self.get_anomaly_sequences(data_copy[f"A_Pred_Global"].values),
                "true": self.get_anomaly_sequences(data_copy["A_True_Global"].values),
            }

            y_min = 1.1 * plot_values["y_true"].min()
            y_max = 1.1 * plot_values["y_true"].max()

            # plot_pred_y = []
            # for i in range(len(anomaly_sequences["pred"])):
            #     tmp = []
            #     for i in range(anomaly_sequences["pred"][i][-1] - anomaly_sequences["pred"][i][0] + 1):
            #         tmp.append(y_max)
            #     plot_pred_y.append(tmp)
            # plot_pred_y = np.array(plot_pred_y)

            y_true_shapes = self.create_shapes(anomaly_sequences["true"], "true", y_min, y_max, plot_values, is_test=is_test)
            # y_pred_shapes = self.create_shapes(anomaly_sequences["pred"], "pred", y_min, y_max, plot_values, is_test=is_test)

            y_df = pd.DataFrame(
                {
                    "timestamp": plot_values["timestamp"].reshape(-1, ),
                    "y_recon": plot_values["y_recon"].reshape(-1, ),
                    "y_true": plot_values["y_true"].reshape(-1, )
                }
            )

            data_type = "Test data" if is_test else "Train data"
            y_true_layout = {
                "title": f"{data_type} | True value for {self.pred_cols[i] if self.pred_cols is not None else ''} and True Label",
                "showlegend": True,
            }

            y_true_layout["shapes"] = y_true_shapes


            lines = [
                go.Scatter(
                    x=y_df["timestamp"],
                    y=y_df["y_true"],
                    line_color="rgb(0, 204, 150, 0.5)",
                    name="y_true",
                    line=dict(width=1)),
                go.Scatter(
                    x=y_df["timestamp"],
                    y=y_df["y_recon"],
                    line_color="rgb(31, 119, 180, 1)",
                    name="y_recon",
                    line=dict(width=1)),
                go.Bar(
                        x=[i for i in range(data_copy[f"A_Pred_Global"].values.shape[0])],
                        y=data_copy[f"A_Pred_Global"].values,
                        width=1.0,
                        # align='center',
                        marker_color='rgb(128,0,128)'
                    )
            ]
            # for i in range(len(plot_pred_y)):
            #     bar = go.Bar(
            #         x=np.array(anomaly_sequences["pred"][i]),
            #         y=np.array(plot_pred_y[i]),
            #         width=2.0,
            #         # align='center',
            #         marker_color='rgb(128,0,128)'
            #     )
            #     lines.append(bar)

            fig = go.Figure(data=lines, layout=y_true_layout)
            py.offline.plot(fig, filename=output_filename)


def get_data_dim(dataset):
    """
    :param dataset: Name of dataset
    :return: Number of dimensions in data
    """
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1