from flask import Flask, render_template
import os

app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'visitors')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER


@app.route("/")
def Display_IMG():
    IMG_LIST = os.listdir('static/visitors')
    IMG_LIST = ['visitors/' + i for i in IMG_LIST]
    return render_template("index.html", imagelist=IMG_LIST)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)