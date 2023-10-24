# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM beasteers/egohos
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
# RUN pip install https://www.piwheels.org/simple/mmcv-full/mmcv_full-1.6.1-py2.py3-none-any.whl#sha256=b54a8eee795abb3202a5bd4d8cac17104147fa0ada8acc4137398e5a5f30367b
# RUN pip install --no-cache-dir https://www.piwheels.org/simple/mmcv-full/mmcv_full-1.7.1-py2.py3-none-any.whl#sha256=5f4d3717c3b2d01b7f070b622a2b4a7ac023fe2cd8eb7d1b8beb99ae4847b8db
RUN pip install --no-cache-dir https://www.piwheels.org/simple/mmcv-full/mmcv_full-1.6.0-py2.py3-none-any.whl#sha256=c1f52964803b152f0debe89fd37d986990bb6a872e4d3776a3d04473fe817404
RUN pip install --no-cache-dir 'mmcls>=0.20.1'

ADD perception/EgoHOS ./EgoHOS
RUN pip install --no-cache-dir --no-deps -e ./EgoHOS
ADD perception/Detic ./Detic
RUN pip install --no-cache-dir -e ./Detic
ADD perception/XMem ./XMem
RUN pip install --no-cache-dir -e ./XMem

RUN pip install --no-cache-dir git+https://github.com/VIDA-NYU/ptgctl.git
ADD perception/requirements.txt .
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/beasteers/redis-record.git

ADD perception/object-states/setup.py perception/object-states/README.md ./object-states/
ADD perception/object-states/object_states/__init__.py ./object-states/object_states/
RUN pip install --no-cache-dir -e ./object-states
ADD perception/object-states ./object-states


# ADD ptgprocess/ $DIR/ptgprocess/
ADD perception/*.py ./

ENTRYPOINT ["python"]
CMD ["main_states.py", "run"]
