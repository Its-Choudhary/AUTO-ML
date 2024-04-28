from flask import Flask, request

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    
    print("Request data:", request.data)
    
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    file.save(file.filename)
    return 'File uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)
