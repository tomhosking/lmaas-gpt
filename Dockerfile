FROM alpine:3.8
MAINTAINER Tom Hosking "quizzl@tomho.sk"

ENV PYTHONUNBUFFERED=1

# RUN apt-get update -y
RUN apk add --no-cache --update python3 py3-pip python3-dev build-base
RUN apk add git
RUN echo "|--> Updating" \
    && apk update && apk upgrade \
    && echo "|--> Install PyTorch" \
    && git clone --recursive https://github.com/pytorch/pytorch \
    && cd pytorch && python setup.py install \
    && echo "|--> Install Torch Vision" \
    && git clone --recursive https://github.com/pytorch/vision \
    && cd vision && python setup.py install \
    && echo "|--> Cleaning" \
    && rm -rf /pytorch \
    && rm -rf /root/.cache \
    && rm -rf /var/cache/apk/* \
    && apk del .build-deps \
    && find /usr/lib/python3.6 -name __pycache__ | xargs rm -r \
    && rm -rf /root/.[acpw]*
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt
ADD . /app
WORKDIR /app/src
ENTRYPOINT ["python3"]
CMD ["-u", "app.py"]