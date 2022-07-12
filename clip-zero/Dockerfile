FROM ptgctl

WORKDIR /src/app

ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN python -c 'import clip;clip.load("ViT-B/32")'

ADD main.py .
CMD ["main.py"]
