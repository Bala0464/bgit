from flask import Flask, render_template, request, send_file
from flask_mail import Mail, Message  # For contact form email functionality
import os
import cv2
import numpy as np
import requests
from skimage.feature import graycomatrix, graycoprops
import pickle
from fpdf import FPDF  # For generating PDF reports

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask-Mail configuration (update with your email credentials)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'balakpgoddatgit ini@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'Goddati@0464'      # Replace with your app-specific password
app.config['MAIL_DEFAULT_SENDER'] = 'your-email@gmail.com'
mail = Mail(app)

# OpenWeatherMap API key
API_KEY = "e4769bd75e3b6170198a9728fc4b419b"  # Replace with your actual API key

# Load crop prediction model and scalers (ensure these files exist)
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Dummy product list for e-commerce (replace with a database in production)
products = [
    {"id": 1, "name": "Rice Seeds", "price": 10.99, "description": "High-quality rice seeds for optimal yield."},
    {"id": 2, "name": "Organic Fertilizer", "price": 15.50, "description": "Natural fertilizer for soil enrichment."},
    {"id": 3, "name": "Maize Seeds", "price": 8.75, "description": "Durable maize seeds for all climates."}
]

# Feature extraction functions
def extract_color_features(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean, std = cv2.meanStdDev(lab_image)
    return np.concatenate([mean, std]).flatten()

def extract_texture_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, correlation, energy, homogeneity]

def infer_npk_levels(color_features, texture_features):
    n = color_features[0] * 0.1 + texture_features[0] * 0.05
    p = color_features[1] * 0.2 + texture_features[1] * 0.1
    k = color_features[2] * 0.15 + texture_features[2] * 0.08
    return n, p, k

# Weather data function
def get_weather_data(api_key, location):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {'q': location, 'appid': api_key, 'units': 'metric'}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        temperature = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        rainfall = weather_data.get('rain', {}).get('1h', 0) * 100  # Convert to mm
        return {'temperature': temperature, 'humidity': humidity, 'rainfall': rainfall}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

# Crop prediction function
def predict_crop(N, P, K, temp, humidity, ph, rainfall):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)
    return crop_dict.get(prediction[0], None)

# PDF report generation
def generate_pdf_report(location, n, p, k, weather_data, crop):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Soil Analysis & Crop Prediction Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Location: {location}", ln=True)
    pdf.cell(200, 10, txt=f"Nitrogen (N): {n:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Phosphorus (P): {p:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Potassium (K): {k:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Temperature: {weather_data['temperature']}°C", ln=True)
    pdf.cell(200, 10, txt=f"Humidity: {weather_data['humidity']}%", ln=True)
    pdf.cell(200, 10, txt=f"Rainfall: {weather_data['rainfall']} mm", ln=True)
    pdf.cell(200, 10, txt=f"Recommended Crop: {crop}", ln=True)
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.pdf')
    pdf.output(report_path)
    return report_path

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/project', methods=['GET', 'POST'])
def project():
    if request.method == 'POST':
        location = request.form['location']
        file = request.files['soil_image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image = cv2.imread(filepath)
            if image is None:
                return render_template('project.html', error="Failed to load the image.")
            try:
                image = cv2.resize(image, (128, 128))
                color_features = extract_color_features(image)
                texture_features = extract_texture_features(image)
                n, p, k = infer_npk_levels(color_features, texture_features)
                weather_data = get_weather_data(API_KEY, location)
                if not weather_data:
                    return render_template('project.html', error="Failed to fetch weather data.")
                crop = predict_crop(n, p, k, weather_data['temperature'], weather_data['humidity'], 7.0, weather_data['rainfall'])
                if crop:
                    result = f"{crop} is the best crop to be cultivated in {location}."
                    report_path = generate_pdf_report(location, n, p, k, weather_data, crop)
                else:
                    result = "Could not determine the best crop."
                return render_template('project.html', soil_image=file.filename, location=location, 
                                      n=n, p=p, k=k, weather_data=weather_data, result=result, report_path=report_path)
            except Exception as e:
                return render_template('project.html', error=f"Error processing the image: {e}")
    return render_template('project.html')

@app.route('/download-report')
def download_report():
    report_path = request.args.get('report_path')
    return send_file(report_path, as_attachment=True)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        msg = Message(subject=f"Contact Form Submission from {name}",
                      recipients=['your-email@gmail.com'],  # Replace with your email
                      body=f"Name: {name}\nEmail: {email}\nMessage: {message}")
        try:
            mail.send(msg)
            return render_template('contact.html', success="Message sent successfully!")
        except Exception as e:
            return render_template('contact.html', error=f"Failed to send message: {e}")
    return render_template('contact.html')



if __name__ == '__main__':
    app.run(debug=True)