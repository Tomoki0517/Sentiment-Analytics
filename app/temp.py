from transformers import pipeline 
from transformers import AutoModelForSequenceClassification 
from transformers import BertJapaneseTokenizer 
 
TARGET_TEXT = "誰でもできる感情分析です。簡単であるので、気軽に試してみましょう。"
 
model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment') 
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking') 
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) 
 
print(nlp(TARGET_TEXT))
