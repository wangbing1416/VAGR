# VAGR

Code and datasets of our paper “Aspect-based Sentiment Analysis with Attention-assisted Graph
and Variational Sentence Representation”



## Requirements

- torch==1.4.0
- scikit-learn==0.23.2
- transformers==3.2.0
- cython==0.29.13
- nltk==3.5

To install requirements, run `pip install -r requirements.txt`.



## Preparation

1. Download and unzip GloVe vectors(`glove.840B.300d.zip`) from https://nlp.stanford.edu/projects/glove/ and put it into `VAGR/glove` directory.

2. Prepare vocabulary with:

   `sh VAGR/build_vocab.sh`



## Training

To train the C3DA model, run:

`sh VAGR/run.sh`



## Logs

Logs are saved under `VAGR/VAGR/log`



## Credits

The code and datasets in this repository are based on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) and [CDT_ABSA](https://github.com/Guangzidetiaoyue/CDT_ABSA).


## Cite

```
@article{feng2022aspect,
  author    = {Shi Feng and Bing Wang and Zhiyao Yang and Jihong Ouyang},
  title     = {Aspect-based sentiment analysis with attention-assisted graph and variational sentence representation},
  journal   = {Knowledge-Based Systems},
  volume    = {258},
  pages     = {109975},
  year      = {2022},
  url       = {https://doi.org/10.1016/j.knosys.2022.109975},
  doi       = {10.1016/j.knosys.2022.109975},
  timestamp = {Wed, 16 Nov 2022 21:55:11 +0100},
  biburl    = {https://dblp.org/rec/journals/kbs/FengWYO22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


