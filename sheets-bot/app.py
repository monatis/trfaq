from __future__ import print_function

import os
import time

from flask import Flask, request
import numpy as np
import tensorflow as tf
from googleapiclient.discovery import build
from transformers import AutoTokenizer, TFAutoModel


# The ID and range of a public spreadsheet.
# Source spreadsheet: https://docs.google.com/spreadsheets/d/1le9MZhI9jJX0QQ4BqIt1_P4OR9Wn1xwOJlp_iG805I4/edit
SPREADSHEET_ID = os.environ.get('GOOGLE_SHEET_ID', None)
if SPREADSHEET_ID is None:
    raise ValueError("You must export GOOGLE_SHEET_ID as an env variable")

 

# DO NOT EDIT THESE LINES.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
model_name = 'mys/bert-base-turkish-cased-nli-mean-faq-mnr'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModel.from_pretrained(model_name)


class SheetsRetriever:
    def __init__(self, sheet_id, range_name='Sheet1!A2:B', interval=5):
        self.sheet_id = sheet_id
        self.range_name = range_name
        self.questions = []
        self.answers = []
        self.last_refresh = -1
        service = build('sheets', 'v4')
        self.sheet = service.spreadsheets()
    
    def fetch_faq_data(self):
        # Call the Sheets API
        result = sheet.values().get(spreadsheetId=self.sheet_id,
                                    range=self.range_name).execute()
        values = result.get('values', [])

        if not values:
            print('No data found.')
        else:
            self.questions, self.answers = self.dispatch_get_questions_and_answers(values)
    
    def dispatch_get_questions_and_answers(values):
        questions = []
        answers = []
        for question, answer in values:
            questions.append("<Q>" + question)
            answers.append("<A>" + answer)

        return questions, answers

    def get(self):
        if time.time() - (interval * 1000) > self.last_refresh:
            self.fetch_faq_data()
            self.last_refresh = time.time()

        return self.questions, self.answers


def answer_faq(model, tokenizer, questions, answers, return_similarities=False):
    q_len = len(questions)
    tokens = tokenizer(questions + answers, padding=True, return_tensors='tf')
    embs = model(**tokens)[0]

    attention_masks = tf.cast(tokens['attention_mask'], tf.float32)
    sample_length = tf.reduce_sum(attention_masks, axis=-1, keepdims=True)
    masked_embs = embs * tf.expand_dims(attention_masks, axis=-1)
    masked_embs = tf.reduce_sum(masked_embs, axis=1) / tf.cast(sample_length, tf.float32)
    a = tf.math.l2_normalize(masked_embs[:q_len, :], axis=1)
    b = tf.math.l2_normalize(masked_embs[q_len:, :], axis=1)

    similarities = tf.matmul(a, b, transpose_b=True)
        
    scores = tf.nn.softmax(similarities)
    results = list(zip(answers, scores.numpy().squeeze().tolist()))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    sorted_results = [{"answer": answer.replace("<A>", ""), "score": f"{score:.4f}"} for answer, score in sorted_results]
    return sorted_results


            
if __name__ == '__main__':
    db = SheetsRetriever(SPREADSHEET_ID)
    
    app = Flask(__name__)

    @app.route('/')
    def index():
        return """
        <h2>FAQ chatbot with Google Sheets and BERT in Turkish</h2>
        <p>A fancy UI coming soon</p>
        <hr />
        <ul id="chat-history">
        </ul>
        <h /r>
        <form action="javascript:void get_answer();">
        <input type="text" name="q" id="q" placeholder="Type something..." />
        <input type="submit" value="✉️" />
        </form>
        <script type="text/javascript">
        function get_answer() {
            q = document.getElementById('q');
            q.disabled = true;
            console.log(q.value);
            hist = document.getElementById('chat-history');
            hist.innerHTML += "<li>🗣️ " + q.value + "</li>";
            fetch('/chat?q=' + q.value).then(function(res) {
                return res.json();
            }).then(function(data) {
                hist.innerHTML += "<li>🤖 " + data['answer'] + "</li>";
                q.value = "";
                q.disabled = false;
            }).catch(function(err) {
                hist.innerHTML += "<li>😭 Bir hata oluştu, console'a bak.</li>";
                q.value = "";
                q.disabled = false;
            });
        }
        </script>
        """


    @app.route('/chat')
    def chat():
        questions, answers = db.get()
        question = request.args.get('q', '')
        if question == '':
            return {"answer": "Merhaba, size nasıl yardımcı olabilirim?", "score": "1.0000"}
        else:
            results = answer_faq(model, tokenizer, [question], answers, return_similarities=True)
            return results[0]


    app.run()