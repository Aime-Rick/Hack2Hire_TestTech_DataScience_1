FROM python:latest

WORKDIR /root/Project/Docker

COPY . .

RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

CMD ["python", "/root/Project/Docker/credit_scoring_prediction.py"]
