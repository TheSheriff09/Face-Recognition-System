# Face Recognition System
A real-time face recognition system that can:

Detect and recognize faces from webcam

Enroll new faces through an interactive interface

Save and load face encodings for persistent recognition

# Features
� Easy enrollment of new people through webcam

⚡ Real-time recognition with adjustable tolerance

📁 Dataset management with automatic folder creation

📊 Performance optimized with configurable settings

📸 Snapshot capability to save recognition results

# Installation
1. Clone the repository:

git clone https://github.com/TheSheriff09/face-recognition-system.git

cd face-recognition-system

2. Install dependencies:

pip install -r requirements.txt

3. Create dataset structure:

mkdir dataset

# Usage
1. Enroll New People:

python face_reco.py enroll --dataset ./dataset

* Press n to create a new person folder

* Press SPACE to capture face images (20 by default)

* Press c to complete current person

* Press q to quit

2. Prepare Encodings:

python face_reco.py prepare --dataset ./dataset --out encodings.pkl

3. Run Recognition:

python face_reco.py camera --encodings encodings.pkl

* Press s to save snapshot

* Press q to quit
