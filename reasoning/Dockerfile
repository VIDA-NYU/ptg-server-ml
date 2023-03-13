FROM ptgctl:gpu

MAINTAINER "rlopez@nyu.edu"

WORKDIR /src/app

ENV REASONING_MODELS_PATH=/src/app/models

RUN apt-get update && apt-get -y install unzip && rm -rf /var/lib/apt/lists/*

# Install tim-reasoning and dependencies
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD tim-reasoning/setup.py tim-reasoning/setup.py

ADD tim-reasoning/setup.py tim-reasoning/README.md tim-reasoning/requirements.txt tim-reasoning/
ADD tim-reasoning/tim_reasoning/__init__.py tim-reasoning/tim_reasoning/
RUN pip install --no-cache-dir -U -e './tim-reasoning'

# Download extra resources
#RUN python -m spacy download en_core_web_lg
#RUN python -c "import nltk;nltk.download('punkt')"

ADD tim-reasoning/tim_reasoning tim-reasoning/tim_reasoning

ADD main.py .
ADD entrypoint.sh .

ENTRYPOINT "/src/app/entrypoint.sh"
#CMD ["main.py", "run" ]
