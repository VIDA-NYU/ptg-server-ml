FROM ptgctl

WORKDIR /src/app

ADD bbn_msgs/requirements.txt .
RUN pip install -r requirements.txt
RUN pip install IPython
ADD bbn_msgs/*.yaml ./
ADD bbn_msgs/*.py ./
CMD ["main.py", "run"]
