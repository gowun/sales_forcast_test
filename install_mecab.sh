#!/bin/bash
# mecab 설치

#curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh
#curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz

javac -version

cd ./mecab
tar zxfv mecab-0.996-ko-0.9.2.tar.gz
#curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar zxvf mecab-ko-dic-2.1.1-20180720.tar.gz

cd mecab-ko-dic-2.1.1-20180720
ldconfig
ldconfig -p | grep /usr/local/lib
cd ..

cd mecab-0.996-ko-0.9.2
./configure
make
make check
make install
cd ..

cp ./nnp.csv mecab-ko-dic-2.1.1-20180720/user-dic/
cd mecab-ko-dic-2.1.1-20180720
#./autogen.sh
./tools/add-userdic.sh
autoreconf
./configure
make
make install
sh -c 'echo "dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic" > /usr/local/etc/mecabrc'