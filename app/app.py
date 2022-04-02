#必要なモジュールをインポート
from flask import Flask, render_template, request
from wtforms import Form, StringField, SubmitField, validators, ValidationError
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
