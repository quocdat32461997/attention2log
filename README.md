# Attention2Log
[Attention2Log](https://github.com/quocdat32461997/attention2log) contains Transformer-basd variants of [DeepLog](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf)'s log key anomaly detection model.

*If you are confusing about how to extract log key (i.e. log template), I recommend using **Drain** which is proposed in this [paper](https://pinjiahe.github.io/papers/ICWS17.pdf). As far as I know, it is the most effective log parsing method. By the way, there is a toolkit and benchmarks for automated log parsing in this [repository](https://github.com/logpai/logparser).*

## Requirement
* python>=3.6
* pytorch==1.4
* tensorboard==2.0.2
* torchmetrics


## Instructions
* **Log Parsing**
```
python3 preprocess.py --input-dir path/to/data-folder --output-dir path/to/save/data
``` 

* **Training**
```
python3 train.py --config configs/transformer_encoder.json --data path/to/processed/data

python3 masked_train.py --config configs/transformer_encoder.json --data path/to/processed/data

```

## Dataset
~~The dataset can be downloaded [HERE](https://www.cs.utah.edu/~mind/papers/deeplog_misc.html).~~ The website can't accessed now, but you can find the HDFS data in this repository.

The original HDFS logs can be found [HERE] (http://people.iiis.tsinghua.edu.cn/~weixu/sospdata.html).

## Visualization
Run the following code in terminal, then navigate to https://localhost:6006.

`tensorboard --logdir=log`

## Reference
Min Du, Feifei Li, Guineng Zheng, Vivek Srikumar. "Deeplog: Anomaly detection and diagnosis from system logs through deep learning." ACM SIGSAC Conference on Computer and Communications Security(CCS), 2017.
