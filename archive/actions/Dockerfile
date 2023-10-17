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
ADD actions/requirements.txt .
RUN pip install -r requirements.txt

ADD ptgprocess/ $DIR/ptgprocess/
ADD actions/main.py .
ADD actions/entrypoint.sh .

#ENTRYPOINT ["bash"]
ENTRYPOINT /src/app/entrypoint.sh
#CMD ["main.py", "run"]
