FROM ptgctl

MAINTAINER "rlopez@nyu.edu"

WORKDIR /src/app

# Install tim-reasoning and dependencies
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Download models
RUN apt-get update; apt-get install unzip
RUN pip install gdown

RUN gdown https://drive.google.com/uc?id=1aYSlngadawRTKuIkd1FtMvrMBfenqPLH -O /src/app/recipe_tagger.zip \
&& unzip /src/app/recipe_tagger.zip -d /src/app \
&& rm /src/app/recipe_tagger.zip

RUN gdown https://drive.google.com/uc?id=1RsXbLrIubPTAbgP3NEAB73LMcAKDV3oE -O /src/app/bert_classifier.zip \
&& unzip /src/app/bert_classifier.zip -d /src/app \
&& rm /src/app/bert_classifier.zip


# Download extra files
RUN python -m spacy download en_core_web_lg

ADD main.py .
CMD ["main.py", "run" ]