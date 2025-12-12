| Dataset                                                      | Train   | Test   | Dimensions(Entity) | Anomalies |
| ------------------------------------------------------------ | ------- | ------ | ------------------ | ------------- |
| [NAB](#Numenta Anomaly Benchmark (NAB))                      |         |        |                    | 0.92          |
| [UCR](#UCR Time Series Anomaly Archive (UCR))                |         |        |                    | 1.88          |
| [SMAP](#Mars Science Laboratory Curiosity Rover Dataset  (MSL)) | 135183  | 427617 | 25 (55)            | 13.13         |
| [MSL](#Mars Science Laboratory Curiosity Rover Dataset  (MSL)) | 58317   | 73729  | 55 (3)             | 10.72         |
| [SWaT](#Secure Water Treatment (SWaT) Dataset)               | 496800  | 449919 | 51 (1)             | 11.98         |
| [WADI](#Water Distribution (WADI) Dataset)                   | 1048571 | 172801 | 123 (1)            | 5.99          |
| [SMD](#Server Machine Dataset (SMD))                         | 708405  | 708420 | 38 (4)             | 4.16          |
| [MSDS](#多源分布式系统（MSDS）数据集)                        | 146430  | 146430 | 10 (1)             | 5.37          |
| [PSM](#Pooled Server Metric (PSM) Dataset)                   |         |        |                    |               |
| [AIOps](#Artificial Intelligence for IT Operations (AIOps) Challenge Datasets) |         |        |                    |               |

### 多源分布式系统（MSDS）数据集
- Repo：https://zenodo.org/records/3549604
- 由复杂分布式系统的分布式跟踪、应用程序日志和指标组成。该数据集专为人工智能操作而构建，包括自动异常检测、根本原因分析和修复。

### Numenta Anomaly Benchmark (NAB)
- Repo：https://github.com/numenta/NAB/tree/master/data ，label在同级文件夹下
- 仓库包含多个数据集

### MIT-BIH 室上性心律失常数据库 (MBA)
- Repo：https://github.com/imperial-qore/TranAD/tree/main/data/MBA
- Description ：MBA是四名患者心电图记录的集合，包含两种不同类型异常（室上性收缩或早搏）的多个实例 

### Secure Water Treatment (SWaT) Dataset
- Repo：填表申请，等待回复邮件https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat
- Direct access：https://drive.google.com/drive/folders/1KOQvV2nR6Y9tIkB4XELHA1CaV1F8LrZ6
- Description：SWaT数据集是生产过滤水的真实工业水处理厂的缩小版。所采集的数据集包含11天连续的运行数据，其中正常运行数据集7天，攻击场景数据集4天，时间间隔1秒。
- train(2015.12.22 16:30:00 -> 2015.12.28 10:00:00). 
- test(2015.12.28 10:00:00->2016.1.2 15:00:00). lebel在test最后一列
- 使用版本：normal有v0和v1两个版本，v1是剔除掉无用的前30分钟数据，故采用normalv1和attackv0
- 原数据中label是normal和attack，统一替换为0和1

### Water Distribution (WADI) Dataset
- Repo：填表申请，等待回复邮件https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#wadi
- Description：该数据集是从WADI测试平台收集的，它是SWaT测试平台的扩展。它包含连续运行16天，其中正常运行收集14天，攻击场景收集2天，时间间隔1秒。
- 使用版本：2019年版本测试集有标签，故使用2019年版本

### Server Machine Dataset (SMD)
- Repo： https://github.com/NetManAIOps/OmniAnomaly
- THU合并原始数据后的数据集，但不知道是如何处理的？：https://drive.google.com/drive/folders/1KOQvV2nR6Y9tIkB4XELHA1CaV1F8LrZ6 （包含MSL SMAP PSM SMD SWaT 5个数据集）
- Paper：KDD 2019  [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network](https://dl.acm.org/doi/10.1145/3292500.3330672).
- SMD（服务器机器数据集）是一个为期 5 周的新数据集。我们从一家大型互联网公司收集了该数据集。该数据集包含 3 组实体。每组实体的名称为 `machine-<group_index>-<index>` ，28个机器实体，每个实体包含38个维度，时间间隔为1分钟。
  **用SMD数据集跑模型的时候，28个机器的数据需要分开训练**。对于每个子集，我们将其分成长度相等的两部分，分别用于训练和测试。我们为每个点添加标签，以区分其是否为异常值，并说明每个异常值对应的维度。
  因此，SMD 由以下部分组成：
  - train：数据集的前半部分。
  - test：数据集的后半部分。
  - test_label：测试集的标签。它表示一个点是否为异常值。
  - interpretation_label: 导致每个异常的维度列表，异常原因。论文中几乎不使用。但TranAD中使用它生成测试机的标签，不是已经有test_label了吗，不理解？
    - 实体级异常，label形状为一列
    - 通道级异常，label形状和test一致
      
- Dataset Information include SMD、MSL、SMAP
  | Dataset name | Number of entities | Number of dimensions | Training set size | Testing set size | Anomaly ratio(%) |
  | ------------ | ------------------ | -------------------- | ----------------- | ---------------- | ---------------- |
  | SMAP         | 55                 | 25                   | 135183            | 427617           | 13.13            |
  | MSL          | 27                 | 55                   | 58317             | 73729            | 10.72            |
  | SMD          | 28                 | 38                   | 708405            | 708420           | 4.16             |

### Soil Moisture Active Passive Satellite Dataset (SMAP) 
### Mars Science Laboratory Curiosity Rover Dataset  (MSL) 
- Repo：https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
- TranAD包含数据和数据预处理：https://github.com/imperial-qore/TranAD/tree/main/data/SMAP_MSL
- Paper：KDD 2018 paper [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/abs/1802.04431)
- Paper Repo：https://github.com/khundman/telemanom
- Info：SMAP和MSL是放在一块的，数据预处理是按照 `labeled_anomalies.csv`区分两个数据集并标注异常
- 土壤湿度主动被动探测卫星 (SMAP) 和火星好奇号 (MSL) 的真实航天器遥测数据和异常值。所有数据均已匿名化处理，时间信息已去除，所有遥测值均根据测试集中的最小值/最大值预先缩放到 `(-1,1)` 之间。通道 ID 也已匿名化处理，但首字母指示通道类型（ `P` = 功率， `R` = 辐射等）。模型输入数据还包含在给定时间窗口内由特定航天器模块发送或接收的命令的独热编码信息。
- 时间间隔1分钟
- labeled_anomalies：数据处理和两个航天器数据分离依靠此文件
  - chan_id: 对应于train和test中对应名字的numpy文件。（对应实体）
  - spacecraft：chan_id所归属的SMAP或MSL
  - anomaly_sequence：对应test中chan_id文件中的index为此范围的为异常序列
  - class：有两种异常，point和contextual，前者点异常，后者是整体变化趋势异常
  - num_values: 对应chan_id的test文件的时间戳数量

### MSCRED synthetic 合成数据集
- Repo：https://github.com/7fantasysz/MSCRED
- Paper：https://arxiv.org/pdf/1811.08055
- TanAD中包含属于预处理步骤，训练集和测试集各10000条，通道级异常标签

### UCR Time Series Anomaly Archive (UCR)
- Repo：https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
- Papers：[The UEA multivariate time series classification archive, 2018 ](https://arxiv.org/abs/1811.00075) and [Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress](https://arxiv.org/abs/2009.13807) 

### Artificial Intelligence for IT Operations (AIOps) Challenge Datasets
- Datasets maintained by the [Netman Lab](https://netman.aiops.org/) at Tsinghua University, their group's GitHub profile can be found [here](https://github.com/NetManAIOps).
- The KPI dataset from their 2018 challenge is [here](https://github.com/NetManAIOps/KPI-Anomaly-Detection), and the 2020 data is [here](https://github.com/NetManAIOps/AIOps-Challenge-2020-Data).

### Pooled Server Metric (PSM) Dataset
- Repo：https://github.com/eBay/RANSynCoders/tree/main/data
- Description：从 eBay 的多个应用程序服务器节点内部收集的，具有 26 个维度

