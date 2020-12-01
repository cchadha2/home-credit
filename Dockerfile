FROM python:3.7

COPY . /src

WORKDIR /src

RUN pip install -r requirements.txt

RUN pip install .

ENTRYPOINT connexion run home_credit/service/api.yml