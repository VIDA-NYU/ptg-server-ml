FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
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
# RUN pip install --no-cache-dir mmcv-full
# RUN pip install --no-cache-dir openmim
# RUN mim install mmcv-full==1.6.1

# ADD perception/EgoHOS ./EgoHOS
# RUN pip install --no-cache-dir -e ./EgoHOS
ADD perception/Detic ./Detic
RUN pip install --no-cache-dir -e ./Detic
ADD perception/XMem ./XMem
RUN pip install --no-cache-dir -e ./XMem

RUN pip install --no-cache-dir git+https://github.com/VIDA-NYU/ptgctl.git
ADD perception/requirements.txt .
RUN pip install -r requirements.txt

ADD perception/object-states/setup.py perception/object-states/README.md ./object-states/
ADD perception/object-states/object_states/__init__.py ./object-states/object_states/
RUN pip install --no-cache-dir -e ./object-states
ADD perception/object-states ./object-states



# ADD ptgprocess/ $DIR/ptgprocess/
ADD perception/*.py ./

ENTRYPOINT ["python"]
CMD ["main_states.py", "run"]
