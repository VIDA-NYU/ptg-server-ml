FROM ptgctl

MAINTAINER "rlopez@nyu.edu"

WORKDIR /src/app

# Install tim-reasoning and dependencies
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Download models
RUN apt-get update; apt-get install unzip
RUN pip install gdown

ENV MODEL_DIR=/src/app/models
RUN mkdir -p /src/app/models

RUN gdown https://drive.google.com/uc?id=1aYSlngadawRTKuIkd1FtMvrMBfenqPLH -O $MODEL_DIR/recipe_tagger.zip \
&& unzip $MODEL_DIR/recipe_tagger.zip -d $MODEL_DIR \
&& rm $MODEL_DIR/recipe_tagger.zip

RUN gdown https://drive.google.com/uc?id=1RsXbLrIubPTAbgP3NEAB73LMcAKDV3oE -O $MODEL_DIR/bert_classifier.zip \
&& unzip $MODEL_DIR/bert_classifier.zip -d $MODEL_DIR \
&& rm $MODEL_DIR/bert_classifier.zip


# Download extra files
RUN python -m spacy download en_core_web_lg
RUN python -c "import nltk;nltk.download('punkt', download_dir='/src/app/models/nltk')"

ADD main.py .
CMD ["main.py", "run" ]

