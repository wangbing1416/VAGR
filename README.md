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





