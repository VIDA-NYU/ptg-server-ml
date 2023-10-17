FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"

WORKDIR /src/app

RUN apt-get update && apt-get install -y python3-opencv git && rm -rf /var/lib/apt/lists/*
RUN apt-get -q update && \
   apt-get install -y build-essential \
   && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir git+https://github.com/VIDA-NYU/ptgctl.git

WORKDIR /src/app

ENV DIR=/src/lib/ptgprocess

#RUN apt-get -q update && \
#    apt-get install -y build-essential \
#    && rm -rf /var/lib/apt/lists/*

#RUN pip install git+https://github.com/NVIDIA/PyProf.git && pip list | grep pyprof
#RUN git clone https://github.com/beasteers/EgoVLP.git # && ls && cd EgoVLP && pip install -r requirements.txt

#ENV DETIC_PATH=/src/app/Detic

ADD setup.py README.md LICENSE $DIR/
ADD ptgprocess/__init__.py $DIR/ptgprocess/
RUN pip install --no-cache-dir -U -e '/src/lib/ptgprocess[yolo,detic,egohos]'

ADD bbn_objects/requirements.txt .
RUN pip install -r requirements.txt

ADD ptgprocess/ $DIR/ptgprocess/
ADD bbn_objects/main.py .

#ENTRYPOINT ["bash"]
CMD ["main.py", "run"]
