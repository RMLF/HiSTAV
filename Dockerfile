FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
MAINTAINER Dr. Daniel Conde, CEris IST-UL

RUN apt-get update && apt-get dist-upgrade -y

# get the tools
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get update && apt-get install -y nano less tree bc
RUN apt-get update && apt-get install -y wget
RUN apt-get update && apt-get install -y clang-format
RUN apt-get update && apt-get install -y doxygen
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y gcc g++
RUN apt-get update && apt-get install -y build-essential
RUN apt-get update && apt-get install -y libboost-all-dev
RUN apt-get update && apt-get install -y cmake
RUN apt-get update && apt-get install -y curl
RUN apt-get update && apt-get install -y unzip

RUN useradd -s /bin/bash -m -r developer
