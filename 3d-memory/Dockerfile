FROM ptgctl:gpu

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
CMD ["main.py", "run"]
