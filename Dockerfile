FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
LABEL maintainer="Sungmin Lee <i.am.sungmin.lee@yonsei.ac.kr>"

# Ubuntu setting
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Installation
RUN \
    apt update -y --fix-missing &&\
    apt install -y sudo

RUN \
    sudo apt install -y apt-utils locales &&\ 
    locale-gen ko_KR.UTF-8

ENV LC_ALL ko_KR.UTF-8
ENV LANG ko_KR.UTF-8
ENV LANGUAGE ko_KR.UTF-8

RUN \
    sudo apt install \
    build-essential \
    clang-format \
    wget \
    software-properties-common \
    -y

RUN \
    sudo apt-get clean &&\
    sudo apt-get autoremove --purge &&\
    sudo apt-get remove python3.10 -y &&\
    sudo apt-get autoremove --purge -y

RUN wget https://bootstrap.pypa.io/get-pip.py

ARG PYTHON_SUBVERSION=3.11
ARG PYTHON_VERSION=${PYTHON_SUBVERSION}.9

# # For Speed Optimization
# RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
# RUN tar -xf Python-${PYTHON_VERSION}.tgz

# RUN \
#     sudo apt install \
#     libffi-dev \
#     zlib1g-dev \
#     libssl-dev \
#     libz-dev \
#     libbz2-dev \
#     libsqlite3-dev \
#     libreadline-dev \
#     libncurses5-dev \
#     libncursesw5-dev \
#     liblzma-dev \
#     libgdbm-dev \
#     libdb5.3-dev \
#     libexpat1-dev \
#     libmpdec-dev \
#     libmpfr-dev \
#     libmpc-dev \    
#     -y

# RUN cd Python-${PYTHON_VERSION} \
#     && ./configure --enable-optimizations \
#     && make -j 48 \
#     && sudo make altinstall

# RUN rm *.tgz
# RUN sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python${PYTHON_SUBVERSION} 1


# For Image size Optimization
RUN sudo apt install -y python${PYTHON_SUBVERSION} python${PYTHON_SUBVERSION}-distutils
RUN sudo update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_SUBVERSION} 1

RUN python get-pip.py
RUN rm *.py

RUN pip install torch
# RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cuml-cu12

RUN \
    pip install \
    numpy==1.26.4 \
    scikit-learn \
    tqdm \
    matplotlib \
    pandas \
    scikit-build \
    requests \
    # einops \
    scipy \ 
    # numba \
    scikit-learn \
    scikit-learn-extra \
    notebook \
    tensorboard \
    # openpyxl \
    ipykernel \
    ipywidgets
    # diffusers \
    # accelerate \
    # transformers

# # OpenCV
# RUN \ 
#     sudo apt install libgl1-mesa-glx libglib2.0-0 -y &&\
#     pip install opencv-python

# # psql
# RUN sudo apt install -y \
#     postgresql-client \
#     curl \
#     git


WORKDIR /workspace