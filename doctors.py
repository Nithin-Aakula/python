doctors = [
    {
        'name': 'Dr. Ramesh',
        'specialization': 'General Physician',
        'id': 'D001',
        'timings': '10:00 AM - 2:00 PM'
    },
    {
        'name': 'Dr. Satish',
        'specialization': 'Neurologist',
        'id': 'D002',
        'timings': '4:00 PM - 8:00 PM'
    },
    {
        'name': 'Dr. Lakshmi',
        'specialization': 'Cardiologist',
        'id': 'D003',
        'timings': '9:00 AM - 1:00 PM'
    }
]


def docs():
    return doctors


def add_doctor(doctors):
    print("\n=== Add New Doctor ===")
    name = input("Enter Doctor Name: ")
    spec = input("Enter Specialization: ")
    did = input("Enter Doctor ID: ")
    timings = input("Enter Available Timings: ")

    new_doc = {
        'name': name,
        'specialization': spec,
        'id': did,
        'timings': timings
    }
    doctors.append(new_doc)
    print("\nDoctor Added Successfully...")


def list_of_Docs(doctors):
    print("\n--- List of Doctors ---")
    if len(doctors) == 0:
        print("No doctors available.")
        return
    for doc in doctors:
        print(f"Name: {doc['name']}, Specialization: {doc['specialization']}, ID: {doc['id']}, Available: {doc['timings']}")


def doctor_available(doctors):
    doc_id = input("Enter Doctor ID to check availability: ")

    for doc in doctors:
        if doc['id'] == doc_id:
            print(f"\n--Doctor Found--")
            print(f"Name: {doc['name']}")
            print(f"Specialization: {doc['specialization']}")
            print(f"Timings: {doc['timings']}")
            return

    print("**Doctor Not Found**")
