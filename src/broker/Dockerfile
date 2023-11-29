FROM python:3.10.11
WORKDIR /app
COPY . /app

RUN apt -y update
RUN apt install -y build-essential cmake pkg-config
RUN apt install -y libx11-dev libatlas-base-dev
RUN apt install -y libgtk-3-dev libboost-python-dev

ENV rmq-server 192.168.1.14
ENV rmq-source-exchange queue.model.input
ENV rmq-completed-exchange exchange.model.output

RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python", "client.py"]
