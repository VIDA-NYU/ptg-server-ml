FROM ptgctl

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# install yolo dependencies and download weights
RUN pip install --no-cache-dir -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
RUN python -c "import torch;torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)"

ADD main.py .
CMD [ "main.py", "run" ]
