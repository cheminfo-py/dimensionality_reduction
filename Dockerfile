
FROM python:3.7-slim-buster

COPY install_packages.sh .
RUN ./install_packages.sh

RUN useradd cheminfo

WORKDIR /home/cheminfo

COPY requirements.txt .

COPY dimensionality_reduction ./dimensionality_reduction

COPY README.md .

RUN pip install --no-cache-dir -r requirements.txt

USER cheminfo

CMD gunicorn -w 4 dimensionality_reduction.dimensionality_reduction:app -b 0.0.0.0:$PORT -k uvicorn.workers.UvicornWorker
