

from register import patient_register, show_patients
from doctors import docs, add_doctor, list_of_Docs, doctor_available
from appoinment import book_appointment, cancel_appointment, view_doctor_schedule, patient_history, generate_daily_list, view_all_appointments

def showoptions():
    print('''
      1. Add Doctor 
      2. Register Patient
      3. Book Appointment
      4. Cancel Appointment
      5. View Doctor's Schedule
      6. View Patient History
      7. Generate Daily Appointment List
      8. Check Doctor Availability
      9. View All Appointments
      10. Exit
    ''')

def main():
    doctors_list = docs()  
    while True:
        showoptions()
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            add_doctor(doctors_list)

        elif choice == '2':
            patient_register()

        elif choice == '3':
            book_appointment(doctors_list)

        elif choice == '4':
            cancel_appointment()

        elif choice == '5':
            view_doctor_schedule()

        elif choice == '6':
            patient_history()

        elif choice == '7':
            generate_daily_list()

        elif choice == '8':
            doctor_available(doctors_list)

        elif choice == '9':
            view_all_appointments()

        elif choice == '10':
            print("***Thank you! Vist Again***")
            break

        else:
            print("Invalid choice, please try again.")

if __name__ == '__main__':
    main()
