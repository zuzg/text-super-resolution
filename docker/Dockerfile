FROM pytorch/pytorch
#:1.13.1-cuda11.6-cudnn8-runtime
RUN mkdir /app
COPY ./requirements.txt /app/requirements.txt
# WORKDIR /app
RUN pip install -r /app/requirements.txt
EXPOSE 8888
CMD ["./start.sh"]
