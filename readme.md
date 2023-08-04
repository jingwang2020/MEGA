### MEGA

This is a Pytorch implement of our MEGA: Multi-scale wavElet Graph Autoencoder.

[Multiscale Wavelet Graph AutoEncoder for Multivariate Time-Series Anomaly Detection | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9954430)

![overall](.\overall.jpg)

### Train and Test

This rep contains a complete set of training and testing frameworks, and you need to install the appropriate dependencies before you can use them. 
You can run main_multiwave.py for training and testing.

```
python main_mutiwave.py --save-plot-data --epoch 50
```

More command parameter settings can be viewed in the args.py.

### Data

More datasets and the data details can be found in the [rep](https://github.com/smallcowbaby/OmniAnomaly).

You can download the dataset through this [link](https://pan.baidu.com/s/19B9I1i_7Nop4AX6vOf_RwQ?pwd=ekyw).

#### Dataset Information

| Dataset name | Number of entities | Number of dimensions | Training set size | Testing set size | Anomaly ratio(%) |
| ------------ | ------------------ | -------------------- | ----------------- | ---------------- | ---------------- |
| SMAP         | 55                 | 25                   | 135183            | 427617           | 13.13            |
| MSL          | 27                 | 55                   | 58317             | 73729            | 10.72            |
| SMD          | 28                 | 38                   | 708405            | 708420           | 4.16             |

#### SMAP and MSL

SMAP (Soil Moisture Active Passive satellite) and MSL (Mars Science Laboratory rover) are two public datasets from NASA.

For more details, see: https://github.com/khundman/telemanom

#### SMD

SMD (Server Machine Dataset) is a new 5-week-long dataset. We collected it from a large Internet company. This dataset contains 3 groups of entities. Each of them is named by `machine-<group_index>-<index>`.

SMD is made up by data from 28 different machines, and the 28 subsets should be trained and tested separately. For each of these subsets, we divide it into two parts of equal length for training and testing. We provide labels for whether a point is an anomaly and the dimensions contribute to every anomaly.

Thus SMD is made up by the following parts:

- train: The former half part of the dataset.
- test: The latter half part of the dataset.
- test_label: The label of the test set. It denotes whether a point is an anomaly.
- interpretation_label: The lists of dimensions contribute to each anomaly.

concatenate

### Author & Contributors

- [Shikuan Shao](https://ieeexplore.ieee.org/author/37089687517)
- [Jing Wang](https://ieeexplore.ieee.org/author/37088687078)
- [Yunfei Bai](https://ieeexplore.ieee.org/author/37089687625)
- [Jiaoxue Deng](https://ieeexplore.ieee.org/author/37089255515)
- [Youfang Lin](https://ieeexplore.ieee.org/author/37598449100)
- [Chao Zhong]()

