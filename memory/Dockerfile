FROM ptgctl

WORKDIR /src/app

# Install tim-reasoning and dependencies
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD utils.py .
ADD impl.py .
ADD main.py .
CMD ["main.py", "run" ]