patients = [] 


def patient_register():
    print("\n=== Patient Registration ===")
    name = input("Enter Patient Name: ")
    age = input("Enter Age: ")
    gender = input("Enter Gender: ")
    pid = input("Enter Patient ID: ")
    contact = input("Enter Contact Number (optional): ")

    new_patient = {
        'name': name,
        'age': age,
        'gender': gender,
        'id': pid,
        'contact': contact
    }
    patients.append(new_patient)
    print("\nPatient Registered Successfully....")
    print(f"Name: {name}, ID: {pid}")


def show_patients():
    print("\n--- Registered Patients ---")
    if len(patients) == 0:
        print("No patients registered yet.")
        return
    for p in patients:
        print(f"Name: {p['name']}, Age: {p['age']}, Gender: {p['gender']}, ID: {p['id']}")
