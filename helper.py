import pandas as pd
import random

# 1. Define common Sri Lankan names (Male, Female, and Surnames)
male_first = [
    "Kasun", "Nuwan", "Tharindu", "Lahiru", "Amila", "Chamara", "Dasun",
    "Kavindu", "Pasan", "Heshan", "Sanjaya", "Isuru", "Malith", "Ruwan", "Venusha", "Pathum", "Ravindu", "Roneth", "Bimasha", "Don", "Kalana", "Dasun", "Dilshan", "Sonal", "Lahiru", "Udara", "Menuwan", "Rukshan", "Hashan",
    "Dinesh", "Gayan", "Charith", "Sahan", "Vimukthi", "Supun"
]

female_first = [
    "Nethmi", "Sanduni", "Tharushi", "Kavindi", "Hashini", "Sachini",
    "Piyumi", "Nipuni", "Anuki", "Dewmini", "Hiruni", "Oshadi", "Chamodi",
    "Dilini", "Imesha", "Hansani", "Dinithi", "Kavisha", "Navodi", "Pinidi", "Sithmi", "Vihangi", "Nethma", "Madusha", "Savindi", "Nipuni", "Thurya", "Nethmi", "Thihansa", "Isuri", "Nimesha", "Heshara", "Shanuli", "Sinoli", "Maheshi"
]

last_names = [
    "Perera", "Silva", "Fernando", "Bandara", "Rajapaksha", "Jayasuriya",
    "Ratnayake", "Gunawardena", "Senanayake", "Jayawardena", "Weerasinghe",
    "Dissanayake", "Gamage", "Liyanage", "Fonseka", "Ekanayake", "Wickramasinghe", "Vithanage", "Yatagama", "Warnakulasooriya", "Alwis", "Rodrigo", "Herath", "Pathirana", "madushanka", "Weerasekara", "Wijerathna", "Amarasinghe", "Yapa", "Kulatunga", "Basnayaka", "Edirisinghe", "Hewage", "Kodithuwakku", "De Soyza", "Martis", "Costa", "Gunathilaka", "Hapuarachchi", "Hettiarachchi", "Tennakon", "Dharmawardana", "Premaratna", "Nanayakkara", "Gunasekara", "Gunarathna", "Dias", "Peris", "Rajapaksa"
]

# 2. Calculate exact distribution
total_names = 1211
single_name_count = int(total_names * 0.65) # 787 names
full_name_count = total_names - single_name_count # 424 names

generated_names = []

# 3. Generate 65% First Name Only
for _ in range(single_name_count):
    # Randomly pick male or female
    if random.choice([True, False]):
        generated_names.append(random.choice(male_first))
    else:
        generated_names.append(random.choice(female_first))

# 4. Generate 35% First and Last Name
for _ in range(full_name_count):
    if random.choice([True, False]):
        first = random.choice(male_first)
    else:
        first = random.choice(female_first)

    last = random.choice(last_names)
    generated_names.append(f"{first} {last}")

# 5. Shuffle the list so the single and full names are randomly mixed
random.shuffle(generated_names)

# 6. Add to your existing CSV
# Load your dataset

# IMPORTANT: If your CSV only has 1200 rows, use generated_names[:1200]
# If your CSV actually has 1211 rows now, use generated_names
df.insert(loc=1, column='name', value=generated_names[:len(df)])

# Save the updated dataset
df.to_csv('updated_data.csv', index=False)

print(f"Successfully generated {len(generated_names)} names and updated the CSV.")