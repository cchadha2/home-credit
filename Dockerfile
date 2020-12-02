FROM python:3.7

COPY . /src

WORKDIR /src

RUN pip install .

ENV MODEL_PATH="model.pkl"

CMD connexion run home_credit/service/api.yml