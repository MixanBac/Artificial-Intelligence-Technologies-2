FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install numpy opencv-python timm 

#Labels as key value pair
LABEL Maintainer="mixbix"


# Any working directory can be chosen as per choice like '/' or '/home' etc
WORKDIR /usr/app/src