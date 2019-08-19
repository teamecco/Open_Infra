FROM tensorflow/tensorflow:latest

RUN pip install pandas sklearn

COPY Web\ Server/train.py /train/
#COPY ./data /nfs/data

WORKDIR /train

ENTRYPOINT ["python", "train.py"]
