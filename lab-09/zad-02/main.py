import nltk

# Pobieranie niezbędnych zasobów dla NLTK i TextBlob
# nltk.download('vader_lexicon')
# nltk.download('punkt')

# Pozytywna opinia
review_pos = """
Absolutely wonderful stay! The staff was incredibly helpful and the room was spotless and beautiful. 
The breakfast buffet had a great variety of delicious options. I highly recommend this hotel to anyone visiting the city. 
It was a truly amazing and relaxing experience.
"""

# Negatywna opinia
review_neg = """
Terrible experience. The room was dirty, smelled terribly like smoke, and the air conditioning was completely broken. 
When I complained, the receptionist was incredibly rude, arrogant and unhelpful. I will never stay here again. 
A complete waste of money and a ruined vacation.
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

print("--- Wyniki NLTK VADER ---")
print("Opinia POZYTYWNA:", sia.polarity_scores(review_pos))
print("Opinia NEGATYWNA:", sia.polarity_scores(review_neg))
print("\n")

from textblob import TextBlob

print("--- Wyniki TextBlob ---")
blob_pos = TextBlob(review_pos)
blob_neg = TextBlob(review_neg)

print(f"Opinia POZYTYWNA: Polarity: {blob_pos.sentiment.polarity:.3f}, Subjectivity: {blob_pos.sentiment.subjectivity:.3f}")
print(f"Opinia NEGATYWNA: Polarity: {blob_neg.sentiment.polarity:.3f}, Subjectivity: {blob_neg.sentiment.subjectivity:.3f}")
print("\n")

from nrclex import NRCLex

print("--- Wyniki NRCLex ---")

# 1. Tworzymy puste obiekty
nrc_pos = NRCLex()
nrc_neg = NRCLex()

# 2. Ładujemy tekst nową metodą
nrc_pos.load_raw_text(review_pos)
nrc_neg.load_raw_text(review_neg)

# 3. Wyświetlamy wyniki
print("Opinia POZYTYWNA - dominujące emocje:", nrc_pos.top_emotions)
print("Opinia NEGATYWNA - dominujące emocje:", nrc_neg.top_emotions)
print("\n")

import text2emotion as te

print("--- Wyniki Text2Emotion ---")

# Metoda get_emotion zwraca słownik z wartościami dla 5 głównych emocji (od 0.0 do 1.0)
t2e_pos = te.get_emotion(review_pos)
t2e_neg = te.get_emotion(review_neg)

print("Opinia POZYTYWNA:", t2e_pos)
print("Opinia NEGATYWNA:", t2e_neg)
print("\n")