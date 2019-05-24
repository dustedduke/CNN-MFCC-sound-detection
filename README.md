# CNN-MFCC-sound-detection
Sound detection and classification tool based on mel-frequency cepstral coefficients extraction
(finger snapping sound used as an example)

<p align="center">
  <img src="https://github.com/dustedduke/CNN-MFCC-sound-detection/blob/master/test/Figure_1.png"  width="550" height="450">
</p>

## Install
```shell
1. git clone https://github.com/dustedDuke/CNN-MFCC-sound-detection.git
2. cd CNN-MFCC-sound-detection
3. pip install -r requirements.txt
4. python make_prediction.py <filepath>
```

## Description
* train_model.ipynb - covers steps for model training with theory description
* make_prediction.py - standalone prediction tool with matplotlib output
* config.ini - configuration of make_prediction.py
* models - folder containing saved models
* train.csv - train folder description file
* test.csv - test folder description file
* train - folder containing train dataset 
(can be downloaded from https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_train.zip?download=1)
* test folder containing test dataset 
(can be downloaded from https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip?download=1
