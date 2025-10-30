from register import patients
from doctors import docs as get_doctors  
appointments = []  


def find_patient_by_id(pid):
    for p in patients:
        if p['id'] == pid:
            return p
    return None


def find_doctor_by_id(did):
    doctors = get_doctors()
    for d in doctors:
        if d['id'] == did:
            return d
    return None


def book_appointment(doctors):
    print("\n=== Book Appointment ===")
    pid = input("Enter Patient ID: ")
    did = input("Enter Doctor ID: ")
    date = input("Enter Date (YYYY-MM-DD): ")
    time = input("Enter Time (HH:MM AM/PM or 24:00): ")

    patient = find_patient_by_id(pid)
    if patient is None:
        print("***Patient not found. Please register patient first.***")
        return

    doctor = find_doctor_by_id(did)
    if doctor is None:
        print("***Doctor not found. Check Doctor ID.***")
        return

   
    for ap in appointments:
        if ap['did'] == did and ap['date'] == date and ap['time'] == time:
            print("This slot is already booked for the selected doctor (double booking)...")
            return

    appointments.append({
        'pid': pid,
        'did': did,
        'date': date,
        'time': time
    })
    print("\n*** Appointment Booked Successfully! ***")


def cancel_appointment():
    print("\n=== Cancel Appointment ===")
    pid = input("Enter Patient ID: ")
    did = input("Enter Doctor ID: ")
    date = input("Enter Date (YYYY-MM-DD): ")
    time = input("Enter Time (HH:MM AM/PM or 24:00): ")

    for app in appointments:
        if app['pid'] == pid and app['did'] == did and app['date'] == date and app['time'] == time:
            appointments.remove(app)
            print("\n*** Appointment Cancelled Successfully! ***")
            return

    print("\nAppointment not found. Please check details....")


def view_doctor_schedule():
    did = input("\nEnter Doctor ID: ")
    date = input("Enter Date (YYYY-MM-DD) to view (press Enter to view all dates): ")

    print(f"\n--- Schedule for Doctor ID {did} ---")
    doctor = find_doctor_by_id(did)
    if doctor is None:
        print("Doctor not found...")
        return

    count = 0
    for app in appointments:
        if app['did'] == did:
            if date == "" or app['date'] == date:
                p = find_patient_by_id(app['pid'])
                pname = p['name'] if p is not None else app['pid']
                print(f"Patient: {pname} | Date: {app['date']} | Time: {app['time']}")
                count += 1

    if count == 0:
        print("No Appointments Found!")


def patient_history():
    pid = input("\nEnter Patient ID: ")
    print(f"\n--- Appointment History for Patient ID {pid} ---")
    p = find_patient_by_id(pid)
    if p is None:
        print("Patient not found...")
        return

    count = 0
    for app in appointments:
        if app['pid'] == pid:
            d = find_doctor_by_id(app['did'])
            dname = d['name'] if d is not None else app['did']
            print(f"Doctor: {dname} | Date: {app['date']} | Time: {app['time']}")
            count += 1

    if count == 0:
        print("No Appointments Found!")


def generate_daily_list():
    date = input("\nEnter Date (YYYY-MM-DD) for daily report: ")
    doctors = get_doctors()

    print(f"\n=== TODAYS APPOINTMENTS ===  Date: {date}\n")
    overall_total = 0

    for doc in doctors:
        doc_apps = []
        for app in appointments:
            if app['did'] == doc['id'] and app['date'] == date:
                p = find_patient_by_id(app['pid'])
                pname = p['name'] if p is not None else app['pid']
                doc_apps.append({'time': app['time'], 'pname': pname})

        if len(doc_apps) > 0:
            print(f"Doctor: {doc['name']} ({doc['specialization']})\n")
            index = 1
            for da in doc_apps:
                print(f"{index}. {da['time']} - {da['pname']}")
                index += 1
            print(f"Total Appointments for {doc['name']}: {len(doc_apps)}\n")
            overall_total += len(doc_apps)

    if overall_total == 0:
        print("No Appointments Found for this date.")


def view_all_appointments():
    print("\n--- All Appointments ---")
    if len(appointments) == 0:
        print("No Appointments Added Yet!")
        return

    for app in appointments:
        p = find_patient_by_id(app['pid'])
        pname = p['name'] if p is not None else app['pid']
        d = find_doctor_by_id(app['did'])
        dname = d['name'] if d is not None else app['did']
        print(f"Patient: {pname} | Doctor: {dname} | Date: {app['date']} | Time: {app['time']}")
