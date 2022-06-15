FROM python:3.7

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

COPY dist/ml_project-0.1.0.tar.gz /ml_project-0.1.0.tar.gz
RUN pip install /ml_project-0.1.0.tar.gz

COPY data/ /data
#COPY ml_project/configs /configs
#RUN chmod 777 /ml_project/*.py
RUN mkdir -p /models

WORKDIR /

CMD ["ml_project_train"]
