import pandas as pd
import mysql.connector
import json

# ==============================
# DATABASE CONNECTION
# ==============================
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="sqlRoot@2025",
    database="ML"
)

cursor = conn.cursor()

# ==============================
# CREATE TABLES
# ==============================

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INT PRIMARY KEY,
    age INT,
    gender VARCHAR(20),
    target_gender VARCHAR(20),
    location VARCHAR(100),
    occupation VARCHAR(100),
    anxiety DECIMAL(4,2),
    avoidance DECIMAL(4,2)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS interest_categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS interests (
    interest_id INT AUTO_INCREMENT PRIMARY KEY,
    category_id INT,
    interest_name VARCHAR(100),
    UNIQUE(category_id, interest_name),
    FOREIGN KEY (category_id) REFERENCES interest_categories(category_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_interests (
    user_id INT,
    interest_id INT,
    PRIMARY KEY (user_id, interest_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (interest_id) REFERENCES interests(interest_id)
)
""")

conn.commit()

# ==============================
# LOAD CSV
# ==============================
df = pd.read_csv("rl_dating_environment_v3.csv")

# Interest columns
interest_columns = [
    "Lifestyle",
    "Arts & Creativity",
    "Music",
    "Movies & Shows",
    "Intellectual & Learning",
    "Food & Drinks",
    "Sports & Outdoor",
    "Gaming & Digital",
    "Travel & Culture",
    "Personality & Values",
    "Relationship Intent"
]

# ==============================
# INSERT CATEGORIES
# ==============================
category_map = {}

for col in interest_columns:
    cursor.execute(
        "INSERT IGNORE INTO interest_categories (category_name) VALUES (%s)",
        (col,)
    )
conn.commit()

cursor.execute("SELECT category_id, category_name FROM interest_categories")
for cid, name in cursor.fetchall():
    category_map[name] = cid

# ==============================
# INSERT USERS
# ==============================

for _, row in df.iterrows():
    cursor.execute("""
        INSERT IGNORE INTO users 
        (user_id, age, gender, target_gender, location, occupation, anxiety, avoidance)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        int(row["user_id"]),
        int(row["age"]),
        row["gender"],
        row["target_gender"],
        row["location"],
        row["occupation"],
        float(row["anxiety"]),
        float(row["avoidance"])
    ))

conn.commit()

# ==============================
# INSERT INTERESTS + RELATIONS
# ==============================

interest_cache = {}

for _, row in df.iterrows():
    user_id = int(row["user_id"])

    for col in interest_columns:
        if pd.isna(row[col]):
            continue

        try:
            interests = json.loads(row[col])
        except:
            continue

        category_id = category_map[col]

        for interest in interests:
            key = (category_id, interest)

            # Insert interest if not exists
            if key not in interest_cache:
                cursor.execute("""
                    INSERT IGNORE INTO interests (category_id, interest_name)
                    VALUES (%s, %s)
                """, (category_id, interest))
                conn.commit()

                cursor.execute("""
                    SELECT interest_id FROM interests
                    WHERE category_id=%s AND interest_name=%s
                """, (category_id, interest))
                interest_id = cursor.fetchone()[0]

                interest_cache[key] = interest_id
            else:
                interest_id = interest_cache[key]

            # Insert relation
            cursor.execute("""
                INSERT IGNORE INTO user_interests (user_id, interest_id)
                VALUES (%s, %s)
            """, (user_id, interest_id))

conn.commit()

print("✅ Data successfully imported!")
cursor.close()
conn.close()