FROM ptgctl

ENV DIR=/src/lib/ptgprocess

RUN pip install --no-cache-dir -U torch torchvision

ADD setup.py README.md LICENSE $DIR/
ADD ptgprocess/__init__.py ptgprocess/__version__.py $DIR/ptgprocess/
RUN pip install --no-cache-dir -U -e '/src/lib/ptgprocess[all]'
ADD ptgprocess/ $DIR/ptgprocess/

WORKDIR /src/app
ENTRYPOINT ["bash"]
