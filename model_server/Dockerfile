FROM ptgctl:gpu

WORKDIR /src/app

ENV DIR=/src/lib/ptgprocess

RUN apt-get -q update && \
    apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

ADD setup.py README.md LICENSE $DIR/
ADD ptgprocess/__init__.py $DIR/ptgprocess/
RUN pip install --no-cache-dir -U -e '/src/lib/ptgprocess[all]'

ADD model_server/requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install

ADD ptgprocess/ $DIR/ptgprocess/
ADD model_server/app.py .

WORKDIR /src/app
ENTRYPOINT ["bash"]
CMD ["-c", "serve run app:server"]
