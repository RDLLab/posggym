FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY pyproject.toml ./
COPY setup.py ./
COPY ./posggym/__init__.py /app/posggym/__init__.py

RUN pip install -e .[all]


COPY . .
