FROM ptgctl

MAINTAINER "rlopez@nyu.edu"

WORKDIR /src/app

ENV REASONING_MODELS_PATH=/src/app/models

# Install tim-reasoning and dependencies
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download extra resources
RUN python -m spacy download en_core_web_lg
RUN python -c "import nltk;nltk.download('punkt')"

ADD main.py .
CMD ["main.py", "run" ]
