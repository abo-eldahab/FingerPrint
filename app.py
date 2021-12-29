from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from load_model import prediction


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "this-really-needs-to-be-changed"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg' , 'BMP'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        predict , predict_proba = prediction(image=filename ,model='ANN' if request.form['model'] == "1" else 'CNN')
        flash('Image successfully uploaded')
        return render_template('index.html', filename=filename , predict=predict , predict_proba=[predict_proba])
    else:
        flash('Allowed image types are - png, jpg, jpeg , BMP')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0' ,  debug=True)