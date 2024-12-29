import os
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify, Response
from datetime import date, datetime, timedelta
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import shutil

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.permanent_session_lifetime = timedelta(minutes=30)

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = None

# Initialize directories and files
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time,Active')

def check_user_exists(username, userid):
    if not os.path.exists('static/faces'):
        return False
    for user_folder in os.listdir('static/faces'):
        existing_username = user_folder.split('_')[0]
        existing_userid = user_folder.split('_')[1]
        if username == existing_username or userid == existing_userid:
            return True
    return False

def generate_frames():
    global camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

def identify_face(facearray):
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray)
    except:
        return None

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    if faces:
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')

def get_registered_users():
    users = []
    if os.path.exists('static/faces'):
        for user_folder in os.listdir('static/faces'):
            username = user_folder.split('_')[0]
            userid = user_folder.split('_')[1]
            users.append({'username': username, 'userid': userid})
    return users

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    active_records = df[df['Active'] == 1]
    names = active_records['Name']
    rolls = active_records['Roll']
    times = active_records['Time']
    l = len(active_records)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df[df['Active'] == 1]['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time},1')
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
                         totalreg=len(os.listdir('static/faces')) if os.path.exists('static/faces') else 0,
                         is_admin=session.get('admin', False),
                         registered_users=registered_users)

@app.route('/start_camera')
def start_camera():
    global camera
    camera = cv2.VideoCapture(0)
    return redirect(url_for('take_attendance'))

@app.route('/take_attendance')
def take_attendance():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        flash('No registered users found. Please register first!', 'error')
        return redirect(url_for('home'))
    return render_template('attendance.html')

@app.route('/process_attendance')
def process_attendance():
    global camera
    if camera is None:
        return jsonify({'status': 'error', 'message': 'Camera not initialized'})
    
    ret, frame = camera.read()
    if not ret:
        return jsonify({'status': 'error', 'message': 'Failed to capture image'})
        
    faces = extract_faces(frame)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
        identified_person = identify_face(face.reshape(1, -1))
        
        if identified_person is not None:
            if add_attendance(identified_person[0]):
                camera.release()
                camera = None
                return jsonify({'status': 'success', 'message': 'Attendance marked successfully!'})
            else:
                camera.release()
                camera = None
                return jsonify({'status': 'info', 'message': 'Attendance already marked for today!'})
        else:
            camera.release()
            camera = None
            return jsonify({'status': 'error', 'message': 'User not registered. Please register first!'})
    
    return jsonify({'status': 'continue'})

@app.route('/check_registration', methods=['POST'])
def check_registration():
    username = request.form.get('username')
    userid = request.form.get('userid')
    if check_user_exists(username, userid):
        return jsonify({'exists': True})
    return jsonify({'exists': False})

@app.route('/register', methods=['GET', 'POST'])
def register():
    global camera
    if request.method == 'POST':
        username = request.form['username']
        userid = request.form['userid']
        
        if check_user_exists(username, userid):
            flash('Username or User ID already exists!', 'error')
            return redirect(url_for('register'))
        
        userimagefolder = f'static/faces/{username}_{userid}'
        os.makedirs(userimagefolder)
        
        camera = cv2.VideoCapture(0)
        i = 0
        while i < 5:
            ret, frame = camera.read()
            if not ret:
                break
                
            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.imwrite(f'{userimagefolder}/{i}.jpg', frame[y:y+h, x:x+w])
                i += 1
                
        camera.release()
        camera = None
        
        train_model()
        flash('Registration successful!', 'success')
        return redirect(url_for('home'))
        
    return render_template('register.html')

@app.route('/remove_user/<userid>')
def remove_user(userid):
    if not session.get('admin', False):
        flash('Unauthorized access!', 'error')
        return redirect(url_for('home'))
    
    # Remove user's face data
    for user_folder in os.listdir('static/faces'):
        if user_folder.split('_')[1] == userid:
            shutil.rmtree(f'static/faces/{user_folder}')
            
            # Deactivate user's attendance records
            df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
            df.loc[df['Roll'] == int(userid), 'Active'] = 0
            df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
            
            # Retrain model if there are still users
            if len(os.listdir('static/faces')) > 0:
                train_model()
            else:
                if os.path.exists('static/face_recognition_model.pkl'):
                    os.remove('static/face_recognition_model.pkl')
                    
            flash('User removed successfully!', 'success')
            return redirect(url_for('home'))
            
    flash('User not found!', 'error')
    return redirect(url_for('home'))

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

if __name__ == '__main__':
    app.run(debug=True)