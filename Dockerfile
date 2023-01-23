FROM pytorch/pytorch
#:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

RUN pip install --upgrade pip

# includes all system dependencies of OpenCV
RUN apt-get update && apt-get install -y python3-opencv

COPY . /app

RUN pip install -r /app/requirements.txt

EXPOSE 8888

ENV PYTHONPATH "/app/src"

CMD jupyter lab --port=8888 --no-browser --allow-root --ip=0.0.0.0
