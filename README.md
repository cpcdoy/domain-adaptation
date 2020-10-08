# Domain Adaptation

## Dependencies

There are 2 requirements.txt, one for Ubuntu and another for Windows.

Use as follows: `pip intall -r requirements.txt`

## Pretrained weights

In the save/ directory, I have pretrained weights (se_weights_epoch_0.pth) that achieve :

- Source accuracy: 98 %
- Target accuracy: 97 %
- F1 target: 1.000
- Precision target: 1.000
- Recall target: 1.000

## Python script
### Usage

```Bash
usage: domain_adaptation.py [-h] [--resume RESUME] [--data DATA] [--save SAVE]
                            [--eval EVAL] [--batch_s BATCH_S]
                            [--batch_t BATCH_T] [--disp]

SVHN to MNIST domain adaptation

optional arguments:
  -h, --help         show this help message and exit
  --resume RESUME    Weights path to resume from
  --data DATA        Datasets path
  --save SAVE        Result path
  --eval EVAL        Evaluate model
  --batch_s BATCH_S  Source domain batch size
  --batch_t BATCH_T  Target domain batch size
  --disp             Display predictions during eval
```


## Evaluate      

```Bash
python domain_adaptation.py --resume=./save/se_weights_epoch_0.pth --eval 100 --batch_s 10 --batch_t 10
```

### Evaluate and display samples

```Bash
python domain_adaptation.py --resume=./save/se_weights_epoch_0.pth --eval 1 --disp --batch_s 10 --batch_t 10
```

Close the matplotlib windows (alt-f4 or close button) to get new batch prediction.
Ctrl-C to stop.

## Train

```Bash
python domain_adaptation.py --resume=./save/se_weights_epoch_0.pth --save=./weights/ --batch_s 100  --batch_t 1000
```

Batch sizes have to be big according to the paper, to keep a good distribution of labels/samples (e.g. 100 per class)

### Train from scratch

```Bash
python domain_adaptation.py --save=./weights/ --batch_s 100  --batch_t 1000
```
