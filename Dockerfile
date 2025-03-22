ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        python3.10 \
        python3-pip \
        python3-virtualenv \
        python3-opencv \
        wget \
        vim \
        sudo

RUN apt-get install -y --no-install-recommends \
        python3-dev \
        python3-distutils \
        python3-tk \
        graphviz \
        libgraphviz-dev \
        virtualenv

COPY . /workspace/depth-anything-v2-hailo

WORKDIR /workspace/depth-anything-v2-hailo

RUN virtualenv -p python3.10 .venv

RUN source .venv/bin/activate

RUN pip install --upgrade pip

RUN pip install hailo_dataflow_compiler-3.30.0-py3-none-linux_x86_64.whl

RUN pip install -r requirements.txt

