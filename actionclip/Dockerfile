FROM ptgctl:gpu

WORKDIR /src/app

ENV DIR=/src/lib/ptgprocess

#RUN apt-get -q update && \
#    apt-get install -y build-essential \
#    && rm -rf /var/lib/apt/lists/*


ADD setup.py README.md LICENSE $DIR/
ADD ptgprocess/__init__.py $DIR/ptgprocess/
RUN pip install --no-cache-dir -U -e '/src/lib/ptgprocess'
ADD ptgprocess/ $DIR/ptgprocess/

ADD actionclip/requirements.txt .
RUN pip install -r requirements.txt
ADD actionclip/main.py .

#ENTRYPOINT ["bash"]
CMD ["main.py", "run"]
