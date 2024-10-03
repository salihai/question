from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wikipedia_search')
def wikipedia_search():
   
    return render_template('wikipedia_search.html')

@app.route('/text_search')
def text_search():
    
    return render_template('text_search.html')

if __name__ == '__main__':
    app.run(debug=True)
