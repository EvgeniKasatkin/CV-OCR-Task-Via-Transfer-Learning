FROM python:3.8.0-buster
MAINTAINER Evgeni Kasatkin
ENV TZ=Europe/Moscow

COPY . /recognition
WORKDIR /recognition

RUN pip3 install -r requirements.txt
RUN pip3 install gunicorn
RUN pip3 install flask_wtf


COPY /app .

CMD ["gunicorn", "-b", "0.0.0.0:5000", "server_model:app"]

EXPOSE 5000
