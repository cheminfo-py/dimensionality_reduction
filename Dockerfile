
FROM python:3.8-slim-buster

COPY install_packages.sh .
RUN ./install_packages.sh

COPY requirements.txt .

COPY dimred ./dimred

COPY README.md .

RUN pip install --no-cache-dir -r requirements.txt

CMD gunicorn -w 1  dimred.dimred:app -b 0.0.0.0:$PORT -k uvicorn.workers.UvicornWorker
