from flask import Flask, render_template, request, session
from ensembled_deepfake import download_from_url, process_video_values
import os

os.environ["GIT_TERMINAL_PROMPT"] = "0"

app = Flask("Deepfake_server")
app.secret_key = 'Conazodeclave'

   
# Pagina inicial
@app.route("/")
def home():
    return render_template("index.html")

#Pagina secundaria, donde se introduce la URL y donde se va a quedar pensando
@app.route("/analyzer", methods=["GET", 'POST'])
def analyzer():
    return render_template("analyzer.html")

#Se valida la existencia del video y se prepara el proceso
@app.route("/processing", methods=["GET", "POST"])
def download():
    video_url = request.form.get("video_url")
    video_doc = download_from_url(video_url, out_path=r"G:\UNI\TFG\Web\static\videos")
    if not video_doc:
        return render_template("analyzer_error.html")
    session["video_doc"]=video_doc
    return render_template("processing.html")
    

#Pagina final, donde se enviarán los resultados para mostrarlos por pantalla
@app.route("/results", methods=['GET','POST'])
def analyze():
        
    video_doc = session.get("video_doc")
    
    results = process_video_values(video_doc["file_path"])
    
    return render_template("results.html", results=results, video_doc=video_doc)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)