FROM ptgctl

WORKDIR /src/app

# Install tim-reasoning and dependencies
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD re_id.py .
ADD main.py .
CMD ["main.py", "run" ]