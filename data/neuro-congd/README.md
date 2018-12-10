## Home Directory of Neuro-ConGD Data

This is the place where Neuro ConGD data are placed in order to fit the default path in `../preprocess_events.py` . In total,  prepared data are provided, which are listed in `Neuro-ConGD`.

### Neuro-ConGD
Download Neuro-ConGD data [here](https://pan.baidu.com/s/1oNdTB16QAz3naGJZGhRZ3A) and organize the folders as follows (in `neuro-congd/`):
```
├── train
├─  └── i [80 folders] (1360 gesture instances in totol, will release continously)
├─      ├── events.csv 
├─      ├── labels.csv 
├─      ├── recording-log 
├─      └── segmentation.csv
├── val
├─  └── j [20 folders] (340 gesture instance in total)
├─      ├── events.csv 
├─      ├── labels.csv 
├─      ├── recording-log 
├─      └── segmentation.csv
└── test
    └── k [20 folders] (340 gesture instance in total)
        ├── events.csv 
        ├── recording-log 
        └── segmentation.csv
    
```
In general, the train/val/test ratio is approximatingly set to 4:1:1 and all of the val/test data are released already. The pre-processed will be __uploaded soon__. The users can also pre-processed the data via preprocess_events.py according to their demand.

Please note that the data in subfolders of `train/`, `val/` and `test/` are __continuous__ and __time-ordered__. The `ith` line of `events.csv` correponds to ` ith timestamp` in `data.aedat`. Moreover, if you don't intend to utilize prepared data directly, please download and pre-process the [raw DVS events ]() i.e. `data.aedat` in your favorite methods.
