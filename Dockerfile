FROM python:3.9-slim

WORKDIR /root/Project/Docker

COPY . .

RUN pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

EXPOSE 8501

CMD ["python","-m","streamlit","run", "/root/Project/Docker/app.py"]
