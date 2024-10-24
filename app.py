from flask import Flask, render_template, request, jsonify
import wikipediaapi
from textwrap3 import wrap
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
import numpy as np
import nltk

'''nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('brown')
nltk.download('omw-1.4')'''

from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import traceback
import pke
from flashtext import KeywordProcessor
import locale
locale.getpreferredencoding = lambda: "UTF-8"
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
#from similarity.normalized_levenshtein import NormalizedLevenshtein
import textdistance
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

import spacy

nlp = spacy.load('en_core_web_sm')

from difflib import SequenceMatcher

from itertools import chain


app = Flask(__name__)

allQuestions = []

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
    
    global allQuestions
    
    if request.method == 'POST':
        
        allQuestions = []
        
        selection = request.form.get('selection')  
        if selection:
            #print(selection)
            text = (get_full_text_by_title(selection).split('\n'))[0]
            #print(text)
            
            answers, questions, distractors = generate_question(text)
            
            questions_data = []
            for i in range(len(questions)):
                question_data = {
                    'question': questions[i],
                    'answers': [answers[i]] + distractors[i][:3] 
                }
                
                allQuestions.append(question_data)
                
                
                random.shuffle(question_data['answers'])
                questions_data.append(question_data)
            
            return jsonify({'text': text, 'questions': questions_data})
        
        return jsonify("", "")
    
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
    
    global allQuestions
    
    if request.method == 'POST':
        
        allQuestions = []
        
        query = request.form.get('query')  
        
        if query:
            
            answers, questions, distractors = generate_question(query)
            
            questions_data = []
            for i in range(len(questions)):
                question_data = {
                    'question': questions[i],
                    'answers': [answers[i]] + distractors[i][:3] 
                }
                
                allQuestions.append(question_data)
                
                random.shuffle(question_data['answers'])
                questions_data.append(question_data)
            
            return jsonify({'questions': questions_data})
            
            
    return render_template('text_search.html')


@app.route('/save_question_wikipedia', methods=['POST', 'GET'])
def save_question_wikipedia():
    
    if request.method == 'POST':
        
        print(allQuestions)
        
        return jsonify({'status': True})
    return render_template('wikipedia_search.html')


@app.route('/save_question_text', methods=['POST', 'GET'])
def save_question_text():
    
    if request.method == 'POST':
        
        print(allQuestions)
        
        return jsonify({'status': True})
    return render_template('text_search.html')















def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



#Important Variables

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

summary_model = T5ForConditionalGeneration.from_pretrained("t5-large")
summary_tokenizer = T5Tokenizer.from_pretrained("t5-large")
summary_model = summary_model.to(device)

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)

s2v = Sense2Vec().from_disk('/workspaces/question/s2v_old')

sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')

levenshtein = textdistance.Levenshtein()




def postprocesstext(content):
    final=""
    for sent in sent_tokenize(content): 
        sent = sent.capitalize() 
        final = final + " " + sent
    return final


def summarizer(text, model, tokenizer):
    text = text.strip().replace("\n", " ") 
    text = "summarize:" + text 
    max_len = 512 
    encoding = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False,
                                    truncation=True, return_tensors="pt").to(device) 
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            early_stopping=True,
                            num_beams=3, 
                            num_return_sequences=1, 
                            no_repeat_ngram_size=2, 
                            min_length=75,
                            max_length=300) 
    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs] 
    summary = dec[0]
    summary = postprocesstext(summary)
    summary = summary.strip()

    return summary


def get_nouns_multipartite(content):
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank() 
        
        pos = {'PROPN', 'NOUN'} 
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english') 

        extractor.load_document(input=content,
                                language='en',
                            stoplist=stoplist,
                            normalization=None) 


        extractor.candidate_selection(pos=pos) 

        extractor.candidate_weighting(alpha=1.1,
                                    threshold=0.75,
                                    method='average') 
        keyphrases = extractor.get_n_best(n=15) 

        for val in keyphrases:
            out.append(val[0])

    except:
        out = []
        traceback.print_exc()

    return out


def get_keywords(originaltext, summarytext):
    keywords = get_nouns_multipartite(originaltext) 
    #print("keywords unsummarized: ", keywords)
    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword) 

    keywords_found = keyword_processor.extract_keywords(summarytext) 
    keywords_found = list(set(keywords_found))
    #print("keywords_found in summarized: ", keywords_found)

    important_keywords = []
    for keyword in keywords:
        if keyword in keywords_found:
            important_keywords.append(keyword)

    return important_keywords[:5]


def get_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(text,
                                    max_length=1000,
                                    pad_to_max_length=False,
                                    truncation=True,
                                    return_tensors="pt").to(device)

    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']

    outs = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            early_stopping=True,
                            #num_beans=5,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            max_length=1000)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    Question = dec[0].replace("question:",  "")
    Question = Question.strip()

    return Question




def string_similarity(word1, word2):
    return SequenceMatcher(None, word1, word2).ratio()


def generate_distractors(word, num_distractors=3):
    distractors = []
    
    synsets = wn.synsets(word)
    
    if not synsets:
        return distractors  
    
    lemmas = set(chain(*[syn.lemma_names() for syn in synsets]))
    lemmas.discard(word) 
    
    similar_words = sorted(list(lemmas), key=lambda x: string_similarity(word, x), reverse=True)
    
    distractors = similar_words[:num_distractors]
    
    if len(distractors) < num_distractors:
        distractors += random.sample(list(wn.words()), num_distractors - len(distractors))
    
    return distractors



def generate_question(context):
    summary_text = summarizer(context, summary_model, summary_tokenizer)
    answers = get_keywords(context, summary_text)
    output=""
    questions = []
    distractors = []
    for answer in answers:
        ques = get_question(summary_text, answer, question_model, question_tokenizer)
        questions.append(ques)
        
        distractor = generate_distractors(answer) 
        distractors.append(distractor)
        
        output = output + ques + "\n"
        output = output + "Ans: " + answer.capitalize() + "\n"

        if len(distractor) > 0:
            for dist in distractor[:4]:
                output = output + dist + "\n"

        output = output + "\n"

    return answers, questions, distractors


if __name__ == '__main__':
    set_seed(42)
    app.run(host='0.0.0.0', port=4000, debug=True)
