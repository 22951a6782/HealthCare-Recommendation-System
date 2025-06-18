
# 🩺 **Patient Health Management System**

This is a Flask-based web application that enables patients to manage their health records, perform health checks, chat with a doctor, and view health tips dynamically. Doctors can register, manage assigned patients, and communicate through a secure chat system.

---

## 🌐 **Features**

### 👤 For Patients:
- User registration and secure login with password hashing  
- Interactive chatbot to assist with health-related queries  
- Health check form that evaluates:
  - Blood pressure
  - Heart rate
  - Stress level
  - Exercise time
  - Sleep time
  - Age
- Automatic health tips, symptoms, and risk analysis based on inputs  
- Download latest health report as PDF  
- Visualize personal health trends over time  
- Assign personal doctor by specialization  
- Chat with your assigned doctor  

### 👨‍⚕️ For Doctors:
- Doctor registration and login  
- Dashboard showing their specialization and patients  
- View all assigned patients  
- Chat with individual patients securely  

---

## 🛠️ **Technologies Used**

- **Backend:** Flask (Python)  
- **Frontend:** HTML, Bootstrap (assumed from templates)  
- **Database:** MySQL (via `pymysql`)  
- **Chatbot:** Trained intent model using TensorFlow  
- **PDF Generation:** `xhtml2pdf` (PISA)  
- **Security:** Password hashing via `werkzeug.security`  

---

## 🗃️ **Directory Structure**

```
project/
│
├── templates/                     # HTML templates for rendering
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   └── ... (many others)
├── app.py                         # Main application logic
├── login_intent_model.h5         # Trained chatbot model
├── login_texts.pkl               # Words used in chatbot
├── login_labels.pkl              # Intent labels
├── data.json                     # Chatbot intents and responses
└── README.md                     # This file
```

---

## 🔑 **Setup Instructions**

1. **Install Dependencies**  
   ```bash
   pip install flask pymysql xhtml2pdf tensorflow nltk
   ```

2. **Set Up the Database**
   - Create a MySQL database named `patient_app`
   - Add the following tables:
     - `patients`
     - `doctors`
     - `health_checks`
     - `personal_doctors`
     - `messages`

3. **Prepare Chatbot Files**
   - Place `login_intent_model.h5`, `login_texts.pkl`, `login_labels.pkl`, and `data.json` in the root directory

4. **Run the Application**  
   ```bash
   python app.py
   ```

5. **Access the Application**  
   Open browser at: [http://localhost:5000](http://localhost:5000)

---

## 🧠 **Chatbot Overview**

- The chatbot uses a basic intent classification model to provide responses to common health-related queries.  
- It uses a trained Keras model and NLTK for preprocessing.

---

## 📄 **Health Analysis Logic**

- Ideal health metrics vary by age group.  
- The system provides personalized tips, identifies potential health risks, and symptoms based on input.  
- Results are saved and can be visualized or downloaded as a PDF report.

---

## 🛡️ **Security Features**

- Passwords are stored in a hashed format  
- Session-based authentication for patients and doctors  
- Restricted access to dashboards and communication features  

---

## ✍️ **Future Improvements**

- Add email verification during registration  
- Enable file attachments in doctor-patient chats  
- Add a scheduler/reminder system for health checks  
- Enhance chatbot with more medical intents  

---

## 👨‍⚕️ **Authors and Contributors**

**Developed by:** *Perala Pranitha*  
If you find this useful or would like to contribute, feel free to fork the repo and submit pull requests.
