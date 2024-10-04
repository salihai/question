from flask import Flask, render_template, request, jsonify
#import wikipediaapi
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wikipedia_search', methods=['POST', 'GET'])
def wikipedia_search():
    if request.method == 'POST':
        query = request.form.get('query')  
        if query:
            print(query)
            #search_wikipedia(query)
            #search_results = "aaa"
            #return jsonify(search_results)
        #return jsonify(['a','b'])
    return render_template('wikipedia_search.html')

def search_wikipedia(keyword):
    user_agent = 'MyProjectName (merlin@example.com)'
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent,
              language='en',
              extract_format=wikipediaapi.ExtractFormat.WIKI)
    search_results = wiki_wiki.page(keyword).links.keys()
    return list(search_results)

















@app.route('/text_search', methods=['POST', 'GET'])
def text_search():
    if request.method == 'POST':
        query = request.form.get('query')  
        if query:
            print(query)
    return render_template('text_search.html')










if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
