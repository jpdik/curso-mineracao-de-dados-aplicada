from flask import Flask, render_template, request

import modelo as m

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def buscar():
    noticia = request.form['noticia']
    resposta = {}
    if noticia:
        resposta['precision'] = round(float(m.calcular_noticia(noticia) * 100), 2)
        if resposta['precision'] > 50:
            resposta['target'] = True
        else:
            resposta['target'] = False

    return render_template("index.html", resposta=resposta)

if __name__ == "__main__":
    app.run(debug=True)