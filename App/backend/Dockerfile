# base image
FROM python:3

ENV PYTHONBUFFERED=1

# create a working directory and copy everything to the working directory
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt