# PROJECT 3: NLP & Sentiment Analysis
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

nltk.download('stopwords')

reviews=[
('This movie was absolutely fantastic! Best film of the year.','positive'),
('Terrible acting and boring plot. Complete waste of time!','negative'),
('Loved every minute of it. The director did an amazing job.','positive'),
('Awful storyline. I fell asleep halfway through.','negative'),
('Brilliant performances! The cinematography was breathtaking.','positive'),
('Worst movie I have ever seen. Do not waste your money.','negative'),
('Heartwarming and beautifully crafted. A true masterpiece.','positive'),
('Dull and predictable. The script was painfully bad.','negative')
]

df=pd.DataFrame(reviews,columns=['review','sentiment'])

ps=PorterStemmer()
stop_words=set(stopwords.words('english'))

def clean_text(text):
    text=re.sub('[^a-zA-Z]',' ',text).lower()
    words=text.split()
    words=[ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_review']=df['review'].apply(clean_text)

tfidf=TfidfVectorizer()
X=tfidf.fit_transform(df['clean_review'])
y=df['sentiment']

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.25,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)

pred=model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,pred))

text=' '.join(df['clean_review'])
wc=WordCloud(width=800,height=400).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wc)
plt.axis('off')
plt.title('Word Cloud')
plt.show()

print("Project 3 Completed Successfully!")
