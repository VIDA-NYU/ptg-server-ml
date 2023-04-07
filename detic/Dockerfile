FROM ptgctl:gpu

WORKDIR /src/app

ENV DIR=/src/lib/ptgprocess

RUN apt-get -q update && \
    apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*
#RUN pip install --no-cache-dir -U torch torchvision
#RUN git clone https://github.com/facebookresearch/detectron2.git && cd detectron2 && pip install -e .
#RUN git clone https://github.com/facebookresearch/Detic.git --recurse-submodules && cd Detic && pip install -r requirements.txt
#ENV DETIC_PATH=/src/app/Detic

ADD setup.py README.md LICENSE $DIR/
ADD ptgprocess/__init__.py $DIR/ptgprocess/
RUN pip install --no-cache-dir -U -e '/src/lib/ptgprocess[detic,egohos]'
ADD ptgprocess/ $DIR/ptgprocess/

ADD detic/requirements.txt .
RUN pip install -r requirements.txt
RUN pip install timm # fixme
ADD detic/main.py .
ADD detic/entrypoint.sh .

ENTRYPOINT "/src/app/entrypoint.sh"

#WORKDIR /src/app
#ENTRYPOINT ["bash"]
#CMD ["main.py", "run"]
