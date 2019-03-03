FROM ubuntu:latest
MAINTAINER Tom Hosking "code@tomho.sk"

ENV PYTHONUNBUFFERED=1

RUN apt-get update -y
RUN apt-get install -y python3 python3-pip python3-dev build-essential


COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt
ADD . /app
WORKDIR /app/src
ENTRYPOINT ["python3"]
CMD ["-u", "app.py"]