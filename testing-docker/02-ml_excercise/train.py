import psycopg2, os, time
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# train
print("Inizio il train con Iris Dataset...")
X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier()
clf.fit(X, y)
acc = clf.score(X, y)


# Procedo a fare log su postgres
time.sleep(5)
conn = psycopg2.connect(
    host     = os.environ["DB_HOST"],
    database = os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"]
)

cur = conn.cursor() #questa permette di eseguire comando SQL in python
cur.execute("""
CREATE TABLE IF NOT EXISTS results (
    id SERIAL PRIMARY KEY,
    accuracy FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

cur.execute(
    "INSERT INTO results (accuracy) VALUES (%s);",
    (acc,)
)
conn.commit()
cur.close()
conn.close()
# INSERT accuracy, timestamp 