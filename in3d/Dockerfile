FROM python:3.9-slim
RUN apt-get update && apt-get install -y python3-opencv git && rm -rf /var/lib/apt/lists/*

WORKDIR /src/app

# Install tim-reasoning and dependencies
ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD main.py .
ENTRYPOINT ["python"]
CMD ["main.py", "run" ]
