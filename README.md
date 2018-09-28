## ELMo for NER 

This project is a Tensorflow implementation of "mainstream" neural tagging scheme based on works of [Deep contextualized word representations, Peters, 
et. al., 2018](https://arxiv.org/pdf/1802.05365.pdf). 


### Requirements

- python 3.6
- tensorflow 1.10.0
- numpy 1.14.3
- gensim 3.6.0
- tqdm 4.26.0

### Evaluation

| Model  | Dataset    | Test F1 |
| :----: | :-------:  | :-----: | 
| Peters et. al | CoNLL 2003 | 92.22(+/-0.10)   |
| Ours          | CoNLL 2003 | 92.23   |

### Train

```
python elmo_train.py
```


### Training Log

| Epoch  | Loss   | Dev F1 | Test F1 |
| :----: | :----: | :----: | :-----: | 
| 1      | 32237  | 90.81  | 87.77   |
| 2      | 12320  | 93.21  | 90.16   |
| 3      | 8823   | 94.19  | 91.75   |
| 4      | 6900   | 94.80  | 91.74   |
| 5      | 5821   | 94.30  | 91.03   |
| 6      | 4996   | 94.92  | 91.26   |
| 7      | 4467   | 95.18  | 92.06   |
| 8      | 3869   | 95.06  | 91.54   |
| 9      | 3483   | 95.13  | 91.88   |
| 10     | 3500   | 95.42  | 91.66   |
| 11     | 2989   | 95.01  | 91.82   |
| 12     | 2770   | 95.39  | 91.70   |
| 13     | 2649   | 95.39  | 91.68   |
| 14     | 2529   | 95.44  | **92.23**   |
| 15     | 2407   | 95.02  | 91.77   |
| 16     | 2140   | 95.13  | 91.80   |
| 17     | 2149   | 95.09  | 92.06   |
| 18     | 1935   | 95.22  | 91.58   |
| 19     | 1946   | 94.88  | 91.91   |
| 20     | 1767   | 95.35  | 92.13   |
