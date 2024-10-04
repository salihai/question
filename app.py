from flask import Flask, render_template, request, jsonify
import wikipediaapi

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wikipedia_search')
def wikipedia_search():
    return render_template('wikipedia_search.html')


@app.route('/get_categories', methods=['POST', 'GET'])
def get_categories():
    if request.method == 'POST':
        query = request.form.get('query')  
        if query:
            #print(query)
            queryList = search_wikipedia(query)
            #print(queryList)
            return jsonify(queryList)
        return jsonify([])
    return render_template('wikipedia_search.html')


def search_wikipedia(keyword):
    user_agent = 'MyProjectName (merlin@example.com)'
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent,
              language='en',
              extract_format=wikipediaapi.ExtractFormat.WIKI)
    search_results = wiki_wiki.page(keyword).links.keys()
    return list(search_results)



@app.route('/get_text', methods=['POST', 'GET'])
def get_text():
    if request.method == 'POST':
        selection = request.form.get('selection')  
        if selection:
            #print(selection)
            text = (get_full_text_by_title(selection).split('\n'))[0]
            #print(text)
            return jsonify(text)
        return jsonify("")
    return render_template('wikipedia_search.html')


def get_full_text_by_title(title):
    user_agent = 'MyProjectName (merlin@example.com)'
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent,
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI)
    page = wiki_wiki.page(title)

    if not page.exists():
        return "Makale bulunamadi."
    return page.text


















@app.route('/text_search', methods=['POST', 'GET'])
def text_search():
    if request.method == 'POST':
        query = request.form.get('query')  
        if query:
            print(query)
    return render_template('text_search.html')




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
