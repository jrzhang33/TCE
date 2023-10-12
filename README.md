# TCE

This repository hosts the implementation of the paper entitled "[Temporal Convolutional Explorer Helps Understand 1D-CNN's Learning Behavior in Time Series Classification from Frequency Domain](https://arxiv.org/abs/2310.05467)", which has been accepted as a long paper at CIKM'23. 

## Requirements

The recommended requirements for TCE are specified as follows:

* Python 3.8
* torch==1.13.0
* tsai==0.3.2
* numpy==1.22.4
* pandas==1.5.1
* scikit_learn==1.1.3
* thop==0.1.1
* sktime==0.14.0
* pandas==1.5.1

The dependencies can be installed by:

```bash
pip install -r requirements.txt
```

## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into `datasets/` so that each data file can be located by `datasets/Univariate2018_arff/Univariate_arff/<dataset_name>/<dataset_name>_*.arff`.

* [30 UEA datasets](http://www.timeseriesclassification.com) should be put into `datasets/` so that each data file can be located by `datasets/Multivariate2018_ts/Multivariate_ts/<dataset_name>/<dataset_name>_*.ts`.

* [HAR ](http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)should be  put into `datasets/` so that each data file can be located by `datasets/HAR`.

  




## Usage

- To train and evaluate one model with/without regulatory framework of TCE on a dataset, run the following command:


```train & evaluate
python main.py --loader 'UEA' --dataset 'UWaveGestureLibrary' --regulator True --model "ResNet"
```



- The detailed descriptions about the arguments are as following:

  | Parameter name | Description of parameter                                     |
  | -------------- | ------------------------------------------------------------ |
  | dataset        | The dataset name                                             |
  | loader         | UCR/UEA/HAR                                                  |
  | regulator      | Equip or not equip regulatory framwork                       |
  | model          | ResNet, InceptionTime and FCN                                |
  | skip           | Specify the number of layers to skip (defaults to not be skipped) |
  | filter         | Specify filtered frequency component (defaults to not be filtered) |

  (For descriptions of more arguments, run `python main.py -h`.)

  After training and evaluation, the trained model and evaluation metrics can be found in `./result`. 



## Code description

`TCE/base`: Data preprocessing, basic procedures of training and testing.

`TCE/models`:  1D-CNNs architecture for MTSC and UTSC.

`TCE/regulator`:  Proposed regulatory framework, TCE with focus scale and frequency centroid,  filtering the specified frequency, skipped the the specified layer and training network with our regulatory framework.

## Contact
[junruzhang@zju.edu.cn](mailto:junruzhang@zju.edu.cn)

