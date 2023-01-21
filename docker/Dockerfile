FROM pytorch/pytorch
#:1.13.1-cuda11.6-cudnn8-runtime
RUN mkdir /app
COPY ./requirements.txt /app/requirements.txt
COPY ./start.sh /app/start.sh
# WORKDIR /app
RUN pip install -r /app/requirements.txt
RUN pip install jupyterlab
EXPOSE 8888
CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8888"]
