FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"

WORKDIR /src/app

ENV DIR=/src/app

RUN apt-get update && apt-get install -y python3-opencv git && rm -rf /var/lib/apt/lists/*
RUN apt-get -q update && \
   apt-get install -y build-essential \
   && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir git+https://github.com/VIDA-NYU/ptgctl.git

WORKDIR /src/app

ENV DIR=/src/lib/ptgprocess

RUN apt-get -q update && \
    apt-get install -y libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ADD setup.py README.md LICENSE $DIR/
ADD ptgprocess/__init__.py $DIR/ptgprocess/
RUN pip install --no-cache-dir -U -e '/src/lib/ptgprocess'

ADD steps/requirements.txt .
RUN pip install -r requirements.txt

ADD ptgprocess/ $DIR/ptgprocess/
ADD steps/*.py ./

#ENTRYPOINT ["bash"]
ENTRYPOINT ["python"]
CMD ["main.py", "run"]
