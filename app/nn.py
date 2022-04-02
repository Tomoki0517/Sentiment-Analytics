#必要なモジュールをインポート
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import BertJapaneseTokenizer
import joblib

TARGET_TEXT =input("文字列を入力→")

model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment')
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking') 
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) 

print(nlp(TARGET_TEXT))

joblib.dump(nlp(TARGET_TEXT, "nn.pkl", compress=True))

