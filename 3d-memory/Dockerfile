FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
# 1.13.1-cuda11.6-cudnn8-runtime
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"

WORKDIR /src/app
RUN apt-get update && apt-get install -y python3-opencv git && rm -rf /var/lib/apt/lists/*
RUN apt-get -q update && \
   apt-get install -y build-essential pkg-config \
   && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir git+https://github.com/VIDA-NYU/ptgctl.git

WORKDIR /src/app

RUN apt-get update && \
    apt-get -y install build-essential && \
    rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir orjson opencv-python 'torch<2' torchvision

# build hand_object_detector
ADD 3d-memory/hand_object_detector hand_object_detector
RUN pip install --no-cache-dir -r hand_object_detector/requirements.txt
RUN which g++
RUN env
RUN python -c 'import torch; print(torch.cuda.is_available())'
RUN cd hand_object_detector/lib && python setup.py build develop

# add app code
ADD 3d-memory/*.py ./
ADD main.py .
ENTRYPOINT ["python"]
CMD ["main.py", "run"]
