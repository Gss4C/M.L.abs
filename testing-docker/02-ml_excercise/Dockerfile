FROM python:3.11-slim
WORKDIR /workspace
RUN pip install scikit-learn psycopg2-binary
COPY . .
CMD ["python", "train.py"]