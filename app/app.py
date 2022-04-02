#必要なモジュールをインポート
from flask import Flask, render_template, request
from wtforms import Form, StringField, SubmitField, validators, ValidationError
from transformers import AutoModelForSequenceClassification 
from transformers import BertJapaneseTokenizer
import numpy as np
import joblib

app = Flask(__name__)

@app.route("/")

class SentimentAnalytics(Form):
    Sentence = StringField("文字を入力",
    [validators.InputRequired("この項目は入力必須です")])
    submit = SubmitField("判定")

if __name__ == "__main__":
    app.run()

# 学習モデルを読み込み予測する
def predict(parameters):
    # モデル読み込み
    model = joblib.load('./nn.pkl')
    params = parameters.reshape(1,-1)
    pred = model.predict(params)
    return pred

# ラベルからIrisの名前を取得
def getName(label):
    print(label)
    if label == 0:
        return "Iris Setosa"
    elif label == 1:
        return "Iris Versicolor"
    elif label == 2:
        return "Iris Virginica"
    else:
        return "Error"

app = Flask(__name__)

# Flaskとwtformsを使い、index.html側で表示させるフォームを構築する
class IrisForm(Form):
    SepalLength = StringField("文章を入力してください",
                     [validators.InputRequired("この項目は入力必須です")])


    # html側で表示するsubmitボタンの表示
    submit = SubmitField("判定")

@app.route('/', methods = ['GET', 'POST'])
def predicts():
    form = IrisForm(request.form)
    if request.method == 'POST':
        print('postdesu')
        # if form.validate() == False:
        #     print('こっち行っちゃってる')
        #     return render_template('index.html', form=form)
        # else:
        SepalLength = request.form["SepalLength"]
        model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment') 
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking') 
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        results = nlp(SepalLength)

        return render_template('result.html', label=results[0]['label'], rate=float(results[0]['score']) * 100)
    elif request.method == 'GET':
        print('getdesu')
        return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run()
