FROM ptgctl:gpu

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
RUN pip install --no-cache-dir -U -e '/src/lib/ptgprocess'

ADD perception/requirements.txt .
RUN pip install -r requirements.txt

ADD perception/procedural_step_recog procedural_step_recog
RUN ls -lah && pip install -e ./procedural_step_recog

ADD ptgprocess/ $DIR/ptgprocess/
ADD perception/*.py ./

#ENTRYPOINT ["bash"]
CMD ["main2.py", "run"]
