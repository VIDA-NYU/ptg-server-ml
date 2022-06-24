FROM ptgctl

WORKDIR /src/app

ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

ADD main.py .
CMD ["main.py", "run"]
