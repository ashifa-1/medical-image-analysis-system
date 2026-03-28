FROM python:3.10

WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

COPY . .

RUN pip install -r requirements.txt

CMD ["sh", "run_pipeline.sh"]