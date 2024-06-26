import csv
import random

# List of possible names
names = ["John", "Jane", "Bob", "Alice", "Charlie", "Megan", "David", "Emily"]

# List of possible genders
genders = ["Male", "Female"]

# Open the CSV file in write mode
with open('students.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Grade", "Gender", "Name", "Accepted", "Classes Repeated"])

    # Generate and write the data
    for _ in range(100):  # Generate 100 students
        name = random.choice(names)
        gender = random.choice(genders)
        classes_repeated = random.randint(0, 3)  # Assuming a student can repeat up to 3 classes
        # Grade is inversely proportional to the number of classes repeated
        grade = 10 - classes_repeated
        # If a student has a grade of 10, they are automatically accepted
        accepted = "yes" if grade == 10 else "no"
        writer.writerow([grade, gender, name, accepted, classes_repeated])