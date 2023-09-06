#!/bin/bash
# install python packages 
pip install scikit-learn --use-feature=2020-resolver
pip install mecab-python

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
#sudo pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install tensorflow==2.5.0
pip install keras==2.6.0
pip install nltk
pip install fast-bleu
pip install sentencepiece 
pip install pyjosa
pip install konlpy==0.5.1
pip install mxnet --user
pip install transformers==4.1.1
pip install tokenizers==0.9.4
pip install pytorch-lightning==1.1.0
cp /home/$USER/shared/backup/crystal/kkma-2.0.jar  /opt/conda/lib/python3.8/site-packages/konlpy/java
pip install Werkzeug==2.0.3 --user
pip install skt==0.2.87
pip install datasets
pip install datasets==13.0.0
#sudo pip install google-api-python-client
#sudo pip install google_auth_httplib2

apt-get update
apt-get install -y psmisc