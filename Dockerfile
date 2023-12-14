FROM python:3.9-slim

WORKDIR /app

COPY . /app

ENV RESEARCH_DATA_PATH /app/data

RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

CMD python src/data/make_dataset.py && python src/models/linear_model.py && python src/models/NN_model.py
