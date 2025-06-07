from flask import Flask, render_template, request, redirect, session
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash
from io import StringIO
import csv
from flask import Response
from flask import make_response, render_template
from xhtml2pdf import pisa
from io import BytesIO
from xhtml2pdf import pisa
import io
import subprocess
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model # type: ignore


app = Flask(__name__)
app.secret_key = 'secretkey'

# Database connection
db = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="patient_app"
)
cursor = db.cursor()

# Load chatbot resources
lemmatizer = WordNetLemmatizer()
chatbot_model = load_model('login_intent_model.h5')
chatbot_words = pickle.load(open('login_texts.pkl', 'rb'))
chatbot_classes = pickle.load(open('login_labels.pkl', 'rb'))

with open('data.json') as f:
    chatbot_intents = json.load(f)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word.isalnum()]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, chatbot_words)
    res = chatbot_model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": chatbot_classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints):
    if not ints:
        return "Sorry, I didn't quite understand that. Could you rephrase?"

    tag = ints[0]['intent']
    for i in chatbot_intents['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])
    return "Sorry, I don't have a response for that."


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        phone = request.form['phone']
        dob = request.form['dob']
        gender = request.form['gender']

        sql = "INSERT INTO patients (name, email, password, phone, dob, gender) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (name, email, password, phone, dob, gender))
        db.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password_input = request.form['password']

        cursor.execute("SELECT id, password FROM patients WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user[1], password_input):
          session['user_id'] = user[0]
          return redirect('/dashboard')

        else:
            return "Invalid credentials, try again."
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('patient_dashboard.html', patient_id=session['user_id'])

@app.route('/patient_chat', methods=['POST'])
def patient_chat():
    if 'user_id' not in session:
        return "Session expired. Please login again."

    user_message = request.form['message']
    ints = predict_class(user_message)
    response = get_response(ints)
    return response

@app.route('/health_check', methods=['GET', 'POST'])
def health_check():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        bp = int(request.form['bp_rate'])
        heart = int(request.form['heart_rate'])
        stress = int(request.form['stress_level'])
        exercise = float(request.form['exercise_time'])
        age = int(request.form['age'])
        sleep = float(request.form['sleep_time'])

        # Ideal ranges based on age group
        if 18 <= age <= 30:
            ideal_bp = (90, 120)
            ideal_heart = (60, 100)
            ideal_exercise = 30
            ideal_sleep = (7, 9)
        elif 31 <= age <= 50:
            ideal_bp = (95, 130)
            ideal_heart = (60, 100)
            ideal_exercise = 30
            ideal_sleep = (7, 8)
        else:
            ideal_bp = (100, 140)
            ideal_heart = (60, 100)
            ideal_exercise = 20
            ideal_sleep = (6, 7)

        # General tips bank
        field_tips = {
            "bp_high": ["Limit salt, processed foods", "More potassium-rich foods", "Avoid smoking, alcohol", "Practice breathing exercises"],
            "bp_low": ["Increase water", "Small frequent meals", "Moderate sodium", "Avoid sudden standing"],
            "heart_high": ["Avoid caffeine", "Practice mindfulness", "Moderate exercises", "Avoid overexertion"],
            "heart_low": ["Gradually increase activity", "Stay hydrated", "Avoid sudden activity", "Consult a cardiologist if needed"],
            "stress_high": ["Meditate, do yoga", "Listen to calm music", "Take short work breaks", "Pick up relaxing hobbies"],
            "exercise_low": ["Start walks (10-15 mins)", "Simple home workouts", "Use stairs", "Set step goals"],
            "sleep_low": ["Consistent sleep schedule", "Avoid screens before bed", "Relaxing bedtime routine", "Dark quiet room"],
            "sleep_high": ["Avoid oversleeping", "Stay active daytime", "Limit naps", "Get sunlight exposure"]
        }

        # Dynamic evaluation
        tips = {
            "BP Tips": [],
            "Heart Rate Tips": [],
            "Stress Management": [],
            "Exercise Tips": [],
            "Sleep Tips": []
        }
        risks = []
        symptoms = []

        # BP
        if bp > ideal_bp[1]:
            tips["BP Tips"].extend(field_tips["bp_high"])
            risks.append("Hypertension")
            symptoms.extend(["Headache", "Fatigue", "Chest pain"])
        elif bp < ideal_bp[0]:
            tips["BP Tips"].extend(field_tips["bp_low"])

        # Heart Rate
        if heart > ideal_heart[1]:
            tips["Heart Rate Tips"].extend(field_tips["heart_high"])
            risks.append("High Heart Rate")
            symptoms.extend(["Palpitations", "Dizziness", "Shortness of breath"])
        elif heart < ideal_heart[0]:
            tips["Heart Rate Tips"].extend(field_tips["heart_low"])

        # Stress
        if stress > 5:
            tips["Stress Management"].extend(field_tips["stress_high"])
            risks.append("High Stress")
            symptoms.extend(["Anxiety", "Sweating", "Racing thoughts"])

        # Exercise
        if exercise < ideal_exercise:
            tips["Exercise Tips"].extend(field_tips["exercise_low"])
            risks.append("Low Activity")
            symptoms.extend(["Fatigue", "Weight gain", "Low energy"])

        # Sleep
        if sleep < ideal_sleep[0]:
            tips["Sleep Tips"].extend(field_tips["sleep_low"])
            risks.append("Sleep Deprivation")
            symptoms.extend(["Irritability", "Tiredness", "Difficulty focusing"])
        elif sleep > ideal_sleep[1]:
            tips["Sleep Tips"].extend(field_tips["sleep_high"])
            risks.append("Oversleeping")
            symptoms.extend(["Lethargy", "Brain fog"])

        # Unique values
        risks = list(set(risks))
        symptoms = list(set(symptoms))

        # Save result to DB
        sql = """INSERT INTO health_checks 
                (patient_id, bp_rate, heart_rate, stress_level, exercise_time, age, sleep_time, result) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
        cursor.execute(sql, (session['user_id'], bp, heart, stress, exercise, age, sleep, "Checked"))
        db.commit()

        return render_template("health_result.html",
                               age=age, bp=bp, heart=heart, stress=stress,
                               exercise=exercise, sleep=sleep,
                               ideal_bp=ideal_bp, ideal_heart=ideal_heart,
                               ideal_exercise=ideal_exercise, ideal_sleep=ideal_sleep,
                               tips=tips, risks=risks, symptoms=symptoms)

    return render_template('health_check.html')

@app.route('/health_trends')
def health_trends():
    if 'user_id' not in session:
        return redirect('/login')

    # Fetch patient's health records with additional fields
    cursor.execute("""
        SELECT checked_at, bp_rate, heart_rate, stress_level, exercise_time, sleep_time 
        FROM health_checks 
        WHERE patient_id = %s 
        ORDER BY checked_at
    """, (session['user_id'],))
    records = cursor.fetchall()

    # Organize data into a list of dictionaries
    health_data = [{
        "date": record[0].strftime('%Y-%m-%d'),
        "bp_rate": record[1],
        "heart_rate": record[2],
        "stress_level": record[3],
        "exercise_time": record[4],
        "sleep_time": record[5]
    } for record in records]

    return render_template('health_trends.html', patient_id=session['user_id'], health_data=health_data)


@app.route('/download_health_result')
def download_health_result():
    if 'user_id' not in session:
        return redirect('/login')

    cursor.execute("SELECT * FROM health_checks WHERE patient_id = %s ORDER BY checked_at DESC LIMIT 1", (session['user_id'],))
    record = cursor.fetchone()

    if not record:
        return "No health check found."

    # Unpack data
    bp = record[2]
    heart = record[3]
    stress = record[4]
    exercise = record[5]
    age = record[6]
    sleep = record[7]

    # Generate tips, risks, symptoms
    tips, risks, symptoms, ideal_bp, ideal_heart, ideal_exercise, ideal_sleep = generate_health_analysis(
        bp, heart, stress, exercise, sleep, age
    )

    # Render to a simplified PDF-safe template
    html = render_template('health_result_pdf.html',
                           age=age, bp=bp, heart=heart, stress=stress,
                           exercise=exercise, sleep=sleep,
                           ideal_bp=ideal_bp, ideal_heart=ideal_heart,
                           ideal_exercise=ideal_exercise, ideal_sleep=ideal_sleep,
                           tips=tips, risks=risks, symptoms=symptoms)

    result = BytesIO()
    pisa_status = pisa.CreatePDF(BytesIO(html.encode("utf-8")), dest=result)

    if pisa_status!=0:
        return "Failed to generate PDF"

    response = make_response(result.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=Health_Report.pdf'
    return response


def generate_health_analysis(bp, heart, stress, exercise, sleep, age):
    # Ideal ranges based on age group
    if 18 <= age <= 30:
        ideal_bp = (90, 120)
        ideal_heart = (60, 100)
        ideal_exercise = 30
        ideal_sleep = (7, 9)
    elif 31 <= age <= 50:
        ideal_bp = (95, 130)
        ideal_heart = (60, 100)
        ideal_exercise = 30
        ideal_sleep = (7, 8)
    else:
        ideal_bp = (100, 140)
        ideal_heart = (60, 100)
        ideal_exercise = 20
        ideal_sleep = (6, 7)

    field_tips = {
        "bp_high": ["Limit salt, processed foods", "More potassium-rich foods", "Avoid smoking, alcohol", "Practice breathing exercises"],
        "bp_low": ["Increase water", "Small frequent meals", "Moderate sodium", "Avoid sudden standing"],
        "heart_high": ["Avoid caffeine", "Practice mindfulness", "Moderate exercises", "Avoid overexertion"],
        "heart_low": ["Gradually increase activity", "Stay hydrated", "Avoid sudden activity", "Consult a cardiologist if needed"],
        "stress_high": ["Meditate, do yoga", "Listen to calm music", "Take short work breaks", "Pick up relaxing hobbies"],
        "exercise_low": ["Start walks (10-15 mins)", "Simple home workouts", "Use stairs", "Set step goals"],
        "sleep_low": ["Consistent sleep schedule", "Avoid screens before bed", "Relaxing bedtime routine", "Dark quiet room"],
        "sleep_high": ["Avoid oversleeping", "Stay active daytime", "Limit naps", "Get sunlight exposure"]
    }

    tips = {"BP Tips": [], "Heart Rate Tips": [], "Stress Management": [], "Exercise Tips": [], "Sleep Tips": []}
    risks = []
    symptoms = []

    if bp > ideal_bp[1]:
        tips["BP Tips"].extend(field_tips["bp_high"])
        risks.append("Hypertension")
        symptoms.extend(["Headache", "Fatigue", "Chest pain"])
    elif bp < ideal_bp[0]:
        tips["BP Tips"].extend(field_tips["bp_low"])

    if heart > ideal_heart[1]:
        tips["Heart Rate Tips"].extend(field_tips["heart_high"])
        risks.append("High Heart Rate")
        symptoms.extend(["Palpitations", "Dizziness", "Shortness of breath"])
    elif heart < ideal_heart[0]:
        tips["Heart Rate Tips"].extend(field_tips["heart_low"])

    if stress > 5:
        tips["Stress Management"].extend(field_tips["stress_high"])
        risks.append("High Stress")
        symptoms.extend(["Anxiety", "Sweating", "Racing thoughts"])

    if exercise < ideal_exercise:
        tips["Exercise Tips"].extend(field_tips["exercise_low"])
        risks.append("Low Activity")
        symptoms.extend(["Fatigue", "Weight gain", "Low energy"])

    if sleep < ideal_sleep[0]:
        tips["Sleep Tips"].extend(field_tips["sleep_low"])
        risks.append("Sleep Deprivation")
        symptoms.extend(["Irritability", "Tiredness", "Difficulty focusing"])
    elif sleep > ideal_sleep[1]:
        tips["Sleep Tips"].extend(field_tips["sleep_high"])
        risks.append("Oversleeping")
        symptoms.extend(["Lethargy", "Brain fog"])

    return tips, list(set(risks)), list(set(symptoms)), ideal_bp, ideal_heart, ideal_exercise, ideal_sleep


@app.route('/doctor_register', methods=['GET', 'POST'])
def doctor_register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])  # hashed password
        specialization = request.form['specialization']

        # Insert into DB
        sql = "INSERT INTO doctors (name, email, password, specialization) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (name, email, password, specialization))
        db.commit()

        return redirect('/doctor_login')

    return render_template('doctor_register.html')

@app.route('/doctor_login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'POST':
        email = request.form['email']
        password_input = request.form['password']

        # Fetch doctor from the database
        cursor.execute("SELECT id, name, password, specialization FROM doctors WHERE email = %s", (email,))
        doctor = cursor.fetchone()

        # Debugging: Print fetched doctor details
        print("Fetched doctor:", doctor)  # This will help you see if the query returns the expected result.

        if doctor and check_password_hash(doctor[2], password_input):
            # Password matches, so save session data
            session['doctor_id'] = doctor[0]
            session['doctor_name'] = doctor[1]
            session['specialization'] = doctor[3]
            print(session)  # Debugging session data
            return redirect('/doctor_dashboard')
        else:
            return "Invalid credentials. Please try again."

    return render_template('doctor_login.html')



@app.route('/doctor_dashboard')
def doctor_dashboard():
    if 'doctor_id' not in session:
        return redirect('/doctor_login')

    return render_template('doctor_dashboard.html',
                           doctor_name=session['doctor_name'],
                           specialization=session['specialization'])


@app.route('/logout')
def logout():
    # Check if doctor is logged in and pop doctor session key
    if 'doctor_id' in session:
        session.pop('doctor_id', None)
        return redirect('/doctor_login') 
    
    # Check if patient is logged in and pop patient session key
    if 'user_id' in session:
        session.pop('user_id', None)
        return redirect('/login')  # Redirect to patient login page

    # If no session exists, just redirect to the homepage or a generic login page
    return redirect('/')

@app.route('/get_personal_doctor', methods=['GET', 'POST'])
def get_personal_doctor():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        # If specialization is selected
        if 'specialization' in request.form:
            specialization = request.form['specialization']

            # Fetch doctors with selected specialization
            cursor.execute("SELECT id, name, specialization FROM doctors WHERE specialization = %s", (specialization,))
            doctors = cursor.fetchall()

            if doctors:
                return render_template('choose_doctor.html', doctors=doctors)
            else:
                return "No doctors available for this specialization."

        # If doctor is selected
        elif 'doctor_id' in request.form:
            doctor_id = request.form['doctor_id']
            patient_id = session['user_id']

            # Check if already assigned
            cursor.execute("SELECT * FROM personal_doctors WHERE patient_id = %s AND doctor_id = %s", (patient_id, doctor_id))
            existing_assignment = cursor.fetchone()

            if existing_assignment:
                return "You have already assigned this doctor."

            # If not, assign the doctor
            sql = "INSERT INTO personal_doctors (patient_id, doctor_id) VALUES (%s, %s)"
            cursor.execute(sql, (patient_id, doctor_id))
            db.commit()

            # Fetch doctor details for confirmation page
            cursor.execute("SELECT name, specialization FROM doctors WHERE id = %s", (doctor_id,))
            doctor = cursor.fetchone()

            return render_template('personal_doctor_assigned.html', doctor=doctor)

    return render_template('get_personal_doctor.html')


@app.route('/view_personal_doctor')
def view_personal_doctor():
    if 'user_id' not in session:
        return redirect('/login')

    # Check if patient has an assigned doctor
    cursor.execute("""
        SELECT d.name, d.specialization 
        FROM personal_doctors pd
        JOIN doctors d ON pd.doctor_id = d.id
        WHERE pd.patient_id = %s
    """, (session['user_id'],))
    doctor = cursor.fetchone()

    if doctor:
        return render_template('view_personal_doctor.html', doctor=doctor)
    else:
        return "No personal doctor assigned yet. <a href='/get_personal_doctor'>Assign one now</a>."

@app.route('/chat_with_doctor', methods=['GET', 'POST'])
def chat_with_doctor():
    if 'user_id' not in session:
        return redirect('/login')

    patient_id = session['user_id']

    # Get assigned doctor
    cursor.execute("""
        SELECT d.id, d.name FROM personal_doctors pd
        JOIN doctors d ON pd.doctor_id = d.id
        WHERE pd.patient_id = %s
    """, (patient_id,))
    doctor = cursor.fetchone()

    if not doctor:
        return "No doctor assigned. <a href='/get_personal_doctor'>Assign now</a>"

    doctor_id, doctor_name = doctor

    if request.method == 'POST':
        message = request.form['message']
        sql = "INSERT INTO messages (patient_id, doctor_id, sender, message) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (patient_id, doctor_id, 'patient', message))
        db.commit()

    # Fetch chat history
    cursor.execute("""
        SELECT sender, message, created_at FROM messages
        WHERE patient_id = %s AND doctor_id = %s
        ORDER BY created_at ASC
    """, (patient_id, doctor_id))
    chat_history = cursor.fetchall()

    return render_template('chat_with_doctor.html', doctor_name=doctor_name, chat_history=chat_history)

@app.route('/chat_with_patient/<int:patient_id>', methods=['GET', 'POST'])
def chat_with_patient(patient_id):
    if 'doctor_id' not in session:
        return redirect('/doctor_login')

    doctor_id = session['doctor_id']

    # Get patient details
    cursor.execute("SELECT name FROM patients WHERE id = %s", (patient_id,))
    patient = cursor.fetchone()

    if not patient:
        return "Patient not found."

    patient_name = patient[0]

    if request.method == 'POST':
        message = request.form['message']
        sql = "INSERT INTO messages (patient_id, doctor_id, sender, message) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (patient_id, doctor_id, 'doctor', message))
        db.commit()

    # Fetch chat history
    cursor.execute("""
        SELECT sender, message, created_at FROM messages
        WHERE patient_id = %s AND doctor_id = %s
        ORDER BY created_at ASC
    """, (patient_id, doctor_id))
    chat_history = cursor.fetchall()

    return render_template('chat_with_patient.html', patient_name=patient_name, chat_history=chat_history)


@app.route('/view_assigned_patients')
def view_assigned_patients():
    if 'doctor_id' not in session:
        return redirect('/doctor_login')

    doctor_id = session['doctor_id']

    # Get assigned patients
    cursor.execute("""
        SELECT p.id, p.name FROM personal_doctors pd
        JOIN patients p ON pd.patient_id = p.id
        WHERE pd.doctor_id = %s
    """, (doctor_id,))
    patients = cursor.fetchall()

    return render_template('view_assigned_patients.html', patients=patients)



if __name__ == '__main__':
    app.run(debug=True)
