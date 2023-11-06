FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"

WORKDIR /src/app
RUN apt-get update && apt-get install -y python3-opencv git && rm -rf /var/lib/apt/lists/*
RUN apt-get -q update && \
   apt-get install -y build-essential pkg-config \
   && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir git+https://github.com/VIDA-NYU/ptgctl.git

MAINTAINER "rlopez@nyu.edu"

WORKDIR /src/app

ENV REASONING_MODELS_PATH=/src/app/models
ENV REASONING_LOGS_PATH=/home/ptg/src/storage/logs

# Install tim-reasoning and dependencies
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download extra resources
ADD tim-reasoning tim-reasoning
RUN pip install ./tim-reasoning

ADD main.py .
#ADD entrypoint.sh .
s
ENTRYPOINT ["python"]
CMD ["main.py", "run" ]
