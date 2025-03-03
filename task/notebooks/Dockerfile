
FROM python:3.12-slim
WORKDIR /app

COPY ../../requirements/requirements.txt requirements.txt
COPY app.py app.py
COPY templates/ templates/
COPY IMDB_Dataset.csv IMDB_Dataset.csv

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet omw-1.4

CMD ["python", "app.py"]
