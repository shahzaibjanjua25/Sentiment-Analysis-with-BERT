import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTextEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QFont
from textblob import TextBlob
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification

import torch

class SentimentAnalysisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentiment Analysis")
        self.setGeometry(200, 200, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.text_entry = QTextEdit()
        self.classify_button = QPushButton("Classify")
        self.clear_button = QPushButton("Clear")
        self.result_label = QLabel("")

        self.text_entry.setFont(QFont("Arial", 12))
        self.classify_button.setFont(QFont("Arial", 12))
        self.clear_button.setFont(QFont("Arial", 12))
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))

        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.classify_button.setStyleSheet("QPushButton { background-color: #6495ED; color: white; }")
        self.clear_button.setStyleSheet("QPushButton { background-color: #FF6347; color: white; }")

        self.classify_button.setMinimumHeight(40)
        self.clear_button.setMinimumHeight(40)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.classify_button)
        button_layout.addWidget(self.clear_button)

        layout = QVBoxLayout()
        layout.addWidget(self.text_entry)
        layout.addLayout(button_layout)
        layout.addWidget(self.result_label)

        self.central_widget.setLayout(layout)

        self.classify_button.clicked.connect(self.classify_sentiment)
        self.clear_button.clicked.connect(self.clear_text)

        # Load BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.model.eval()

    def classify_sentiment(self):
        text = self.text_entry.toPlainText()

        # Preprocess the text
        preprocessed_text = self.preprocess_text(text)

        # Perform sentiment analysis using BERT
        sentiment_score = self.analyze_sentiment(preprocessed_text)

        if sentiment_score > 0:
            sentiment = "Positive"
        elif sentiment_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        self.result_label.setText("Sentiment: " + sentiment)

        # Generate sentiment distribution visualization
        self.plot_sentiment_distribution(preprocessed_text)

    def preprocess_text(self, text):
        # Add your preprocessing steps here (e.g., removing stop words, stemming, etc.)
        preprocessed_text = text.lower()
        return preprocessed_text

    def analyze_sentiment(self, text):
        # Tokenize the text
        encoded_input = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        # Perform sentiment analysis using BERT model
        with torch.no_grad():
            output = self.model(**encoded_input)

        scores = output.logits.squeeze()
        sentiment_score = scores.numpy()[0]

        return sentiment_score

    def plot_sentiment_distribution(self, text):
        # Generate sentiment distribution visualization
        blob = TextBlob(text)
        sentiment_scores = [sentence.sentiment.polarity for sentence in blob.sentences]
        labels = ['Positive', 'Negative', 'Neutral']
        sentiment_counts = [len([score for score in sentiment_scores if score > 0]),
                            len([score for score in sentiment_scores if score < 0]),
                            len([score for score in sentiment_scores if score == 0])]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, sentiment_counts)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

    def clear_text(self):
        self.text_entry.clear()
        self.result_label.setText("")



