FROM ptgctl

ENV DIR=/src/lib/ptgprocess

RUN apt-get -q update && \
    apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -U torch torchvision

ADD setup.py README.md LICENSE $DIR/
ADD ptgprocess/__init__.py $DIR/ptgprocess/
RUN pip install --no-cache-dir -U -e '/src/lib/ptgprocess[all]'
RUN pip install --no-cache-dir -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
ADD ptgprocess/ $DIR/ptgprocess/


WORKDIR /src/app
#ENTRYPOINT ["bash"]
