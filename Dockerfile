# Use the base image from NVIDIA, CUDA 11.8, cuDNN 8.9, Ubuntu 22.04
ARG BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

# Install Essential Packages
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

# Install Dependencies for Hailo DFC
RUN apt-get install -y --no-install-recommends \
        python3-dev \
        python3-distutils \
        python3-tk \
        graphviz \
        libgraphviz-dev \
        virtualenv

# Copy the project code to the container
COPY . /workspace/depth-anything-v2-hailo

# Set the working directory
WORKDIR /workspace/depth-anything-v2-hailo

# Create a virtual environment
RUN virtualenv venv

# Upgrade pip
RUN venv/bin/pip install --upgrade pip

# Install Hailo DFC
RUN venv/bin/pip install hailo_dataflow_compiler-3.30.0-py3-none-linux_x86_64.whl

# Install the project dependencies
RUN venv/bin/pip install -r requirements.txt

# Install DALI
RUN venv/bin/pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110 nvidia-dali-tf-plugin-cuda110

# Create a user for the Hailo DFC container
ARG user=hailo
ARG group=hailo
ARG uid=1000
ARG gid=1000
RUN groupadd --gid $gid $group && \
    adduser --uid $uid --gid $gid --shell /bin/bash --disabled-password --gecos "" $user && \
    chmod u+w /etc/sudoers && echo "$user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && chmod -w /etc/sudoers && \
    chown -R $user:$group /workspace