FROM nvidia/cuda:12.0.1-devel-ubuntu20.04
WORKDIR ./h-rocks
RUN apt-get update \
  && apt-get install -y build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev vim git bc \
  && apt-get install -y python3
COPY . .
