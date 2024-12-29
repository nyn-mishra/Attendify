# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from datetime import date, datetime, timedelta
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import shutil

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a random string
app.permanent_session_lifetime = timedelta(minutes=30)

# Admin credentials - change these!
ADMIN_USERNAME = "nayan_mishra"
ADMIN_PASSWORD = "16097264"

# Global variables
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initialize face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize directories
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

def initialize_attendance():
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Roll,Time')

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

def get_face_encoding(img):
    faces = extract_faces(img)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (50, 50))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return face_img.reshape(1, -1)
    return None

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    
    for user in userlist:
        user_images = os.listdir(f'static/faces/{user}')
        for imgname in user_images:
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            face_encoding = get_face_encoding(img)
            if face_encoding is not None:
                faces.append(face_encoding[0])
                labels.append(user)
    
    if faces:
        faces = np.array(faces)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(faces, labels)
        joblib.dump(clf, 'static/face_recognition_model.pkl')
        return True
    return False

def identify_face(frame):
    face_encoding = get_face_encoding(frame)
    if face_encoding is not None:
        try:
            model = joblib.load('static/face_recognition_model.pkl')
            return model.predict(face_encoding)[0]
        except:
            return None
    return None

def check_face_exists(frame):
    face_encoding = get_face_encoding(frame)
    if face_encoding is None:
        return False
    
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        userlist = os.listdir('static/faces')
        if userlist:
            proba = model.predict_proba(face_encoding)
            max_proba = np.max(proba)
            
            if max_proba > 0.7:
                return True
    except:
        pass
    return False

def get_registered_users():
    users = []
    for user_folder in os.listdir('static/faces'):
        username = user_folder.split('_')[0]
        userid = user_folder.split('_')[1]
        users.append({'username': username, 'userid': userid})
    return users

def extract_attendance():
    initialize_attendance()
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    initialize_attendance()
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
        return True
    return False

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    registered_users = get_registered_users() if session.get('admin', False) else []
    return render_template('home.html',
                         names=names,
                         rolls=rolls,
                         times=times,
                         l=l,
                         datetoday2=datetoday2,
                         totalreg=len(os.listdir('static/faces')),
                         is_admin=session.get('admin', False),
                         registered_users=registered_users)

@app.route('/start_attendance')
def start_attendance():
    if not os.listdir('static/faces'):
        flash('No registered users found. Please register first!', 'error')
        return redirect(url_for('home'))

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash('Unable to access camera!', 'error')
            return redirect(url_for('home'))

        attendance_marked = False
        already_marked_users = set()
        max_attempts = 100  # Prevent infinite loop
        attempts = 0

        while not attendance_marked and attempts < max_attempts:
            ret, frame = cap.read()
            if not ret:
                break

            identified_person = identify_face(frame)
            if identified_person:
                if identified_person not in already_marked_users:
                    if add_attendance(identified_person):
                        flash(f'Attendance marked for {identified_person}!', 'success')
                        attendance_marked = True
                    else:
                        already_marked_users.add(identified_person)
                        flash('Attendance already marked for this user today!', 'info')
            
            attempts += 1

        cap.release()
        cv2.destroyAllWindows()

        if not attendance_marked and already_marked_users:
            flash('No new attendance could be marked', 'info')
    
    except Exception as e:
        flash(f'Error during attendance: {str(e)}', 'error')
    
    return redirect(url_for('home'))

@app.route('/check_registration', methods=['POST'])
def check_registration():
    username = request.form.get('username')
    userid = request.form.get('userid')
    
    # Check if username or userid exists
    for user_folder in os.listdir('static/faces'):
        existing_username = user_folder.split('_')[0]
        existing_userid = user_folder.split('_')[1]
        if username == existing_username or userid == existing_userid:
            return jsonify({'exists': True, 'message': 'Username or User ID already exists!'})
    
    return jsonify({'exists': False})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        userid = request.form['userid']
        
        # Check if user already exists
        for user_folder in os.listdir('static/faces'):
            existing_username = user_folder.split('_')[0]
            existing_userid = user_folder.split('_')[1]
            if username == existing_username or userid == existing_userid:
                flash('Username or User ID already exists!', 'error')
                return redirect(url_for('register'))

        try:
            userimagefolder = f'static/faces/{username}_{userid}'
            os.makedirs(userimagefolder)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                flash('Unable to access camera!', 'error')
                return redirect(url_for('register'))
            
            images_to_capture = 10
            captured_images = 0
            
            while captured_images < images_to_capture:
                ret, frame = cap.read()
                if not ret:
                    break

                faces = extract_faces(frame)
                if len(faces) > 0:
                    if check_face_exists(frame):
                        shutil.rmtree(userimagefolder)
                        cap.release()
                        cv2.destroyAllWindows()
                        flash('Face already registered with another user!', 'error')
                        return redirect(url_for('register'))

                    (x, y, w, h) = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    cv2.imwrite(f'{userimagefolder}/face_{captured_images}.jpg', face_img)
                    captured_images += 1
                    
                    cv2.putText(frame, f'Capturing: {captured_images}/{images_to_capture}', 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Capturing Face', frame)
                    cv2.waitKey(500)  # Half-second delay between captures

            cap.release()
            cv2.destroyAllWindows()

            if captured_images < images_to_capture:
                shutil.rmtree(userimagefolder)
                flash('Not enough faces captured. Please try again!', 'error')
                return redirect(url_for('register'))
            # Train model with multiple images
            if train_model():
                flash('Registration successful!', 'success')
            else:
                shutil.rmtree(userimagefolder)
                flash('Error training model. Please try again!', 'error')
                      
        except Exception as e:
            if os.path.exists(userimagefolder):
                shutil.rmtree(userimagefolde)
            flash(f'Error during registration: {str(e)}', 'error')
            return redirect(url_for('register'))
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        if (request.form['username'] == ADMIN_USERNAME and
            request.form['password'] == ADMIN_PASSWORD):
            session['admin'] = True
            flash('Successfully logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials!', 'error')
    return render_template('admin.html')

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('home'))

@app.route('/remove_user/<userid>')
def remove_user(userid):
    if not session.get('admin', False):
        flash('Unauthorized access!', 'error')
        return redirect(url_for('home'))

    try:
        # Remove user's face data
        for user_folder in os.listdir('static/faces'):
            if user_folder.split('_')[1] == userid:
                shutil.rmtree(f'static/faces/{user_folder}')
                
                # Remove user's attendance records
                for attendance_file in os.listdir('Attendance'):
                    file_path = f'Attendance/{attendance_file}'
                    if os.path.isfile(file_path):
                        df = pd.read_csv(file_path)
                        df = df[df['Roll'] != int(userid)]
                        df.to_csv(file_path, index=False)
                
                # Retrain model if there are remaining users
                if os.listdir('static/faces'):
                    train_model()
                elif os.path.exists('static/face_recognition_model.pkl'):
                    os.remove('static/face_recognition_model.pkl')
                
                flash('User removed successfully!', 'success')
                return redirect(url_for('home'))

        flash('User not found!', 'error')
    except Exception as e:
        flash(f'Error removing user: {str(e)}', 'error')
    
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
