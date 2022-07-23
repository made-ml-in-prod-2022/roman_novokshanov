FROM python:3.7

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

COPY dist/ml_project-0.1.0.tar.gz /ml_project-0.1.0.tar.gz
RUN pip install /ml_project-0.1.0.tar.gz

COPY data/ /data
COPY configs/ /configs
RUN mkdir -p /models

WORKDIR /

CMD ["python", "-m", "ml_project.train_pipeline", "--config-path", "/configs"]
