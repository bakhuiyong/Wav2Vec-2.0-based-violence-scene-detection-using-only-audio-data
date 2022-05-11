# Wav2Vec 2.0 based violence scene detection using only audio data

This repository implements Wav2Vec 2.0 based violence scene detection using only audio data.

( http://ieiespc.org/ieiespc/ArticleDetail/RD_R/412815)



## Proposed System

The proposed system is as follows.
The MediaEval2015 audio signal is input to the pre-trained Wav2Vec 2.0. After that, the violent scenes are classified with a classifier.

<img src="/images/model.PNG" width="70%" height="150">

The pre-trained Wav2Vec 2.0 used in the experiment can be downloaded from https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt



## Dataset used in the experiment

[MediaEval2015](https://liris-accede.ec-lyon.fr/) was used for the experiment. 



## Experiment

```
python main.py wav2vec_small.pt media2015(your dataset path) save(your save path)
```



## Reference

1. [Wav2Keyword](https://github.com/qute012/Wav2Keyword)
2. [fairseq_Wav2Vec 2.0](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md)

