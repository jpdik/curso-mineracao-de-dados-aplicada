from flask import Flask, request, Response, render_template, redirect, make_response, url_for
import topic_modeler as tm
import query_sum as qs

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    query = request.args.get('query', '')
    result = []
    if query:
        qs.contar_busca(query)
        result = tm.buscar(query)
    return render_template('index.html', query=query, result=result, top_10=qs.top_10_buscas())

if __name__ == '__main__':
  app.run(host="192.168.1.129", debug=True)