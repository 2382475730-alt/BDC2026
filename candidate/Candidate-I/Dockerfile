FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN python -m pip install --no-cache-dir numpy==2.3.3 pandas==2.3.2

COPY . .
ENV PYTHONHASHSEED=20260708

CMD ["sleep", "infinity"]
