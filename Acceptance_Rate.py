import csv
import random
names = ["John", "Jane", "Bob", "Alice", "Charlie", "Megan", "David", "Emily"]

genders = ["Male", "Female"]

with open('students.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Grade", "Gender", "Name", "Accepted", "Classes Repeated"])

    for _ in range(100):
        name = random.choice(names)
        gender = random.choice(genders)
        classes_repeated = random.randint(0, 3)
        grade = 10 - classes_repeated
        accepted = "yes" if grade == 10 else "no"

        writer.writerow([grade, gender, name, accepted, classes_repeated])
