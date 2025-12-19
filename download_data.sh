#!/bin/bash

mkdir -p data && cd data

curl -L -o lfwpeople.zip \
  https://www.kaggle.com/api/v1/datasets/download/atulanandjha/lfwpeople

unzip lfwpeople.zip && tar xvzf lfw-funneled.tgz
rm -f lfwpeople.zip lfw-funneled.tgz