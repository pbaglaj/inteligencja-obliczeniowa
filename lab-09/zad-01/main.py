import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string

# Pobieranie niezbędnych zasobów NLTK (uruchom to tylko raz)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')

# Wczytanie tekstu
text = open('text.txt', 'r').read()
print(f"Oryginalny tekst ma {len(text.split())} słów (przed tokenizacją).")

# 1. Tokenizacja
# Zamiana na małe litery i tokenizacja
tokens = word_tokenize(text.lower())
print(f"b) Liczba tokenów po tokenizacji: {len(tokens)}")

# 2. Usuwanie stopwords
stop_words = set(stopwords.words('english'))
# Filtrowanie tokenów
filtered_tokens = [word for word in tokens if word not in stop_words]
print(f"c) Liczba tokenów po usunięciu standardowych stop-words: {len(filtered_tokens)}")

# 3. Wlasne stopwords
# Dodanie znaków interpunkcyjnych z modułu string oraz przykładowych słów
custom_stopwords = list(string.punctuation)
custom_stopwords.extend(['``', "''", 'also', 'would', 'could', "u", "'s"]) # NLTK czasem generuje takie cudzysłowy

# Aktualizacja głównej listy (używamy union dla setów lub dodajemy do listy)
all_stopwords = stop_words.union(set(custom_stopwords))

# Ponowne filtrowanie
final_tokens = [word for word in filtered_tokens if word not in all_stopwords]

print(f"d) Liczba tokenów po usunięciu dodatkowych stop-words i interpunkcji: {len(final_tokens)}")

# Lematyzacja
lemmatizer = WordNetLemmatizer()

# Lematyzacja wszystkich przefiltrowanych słów
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in final_tokens]

print("e) Wybrany lematyzer to: WordNetLemmatizer z biblioteki NLTK.")
print(f"   Liczba słów po lematyzacji: {len(lemmatized_tokens)}")
# Uwaga: liczba słów zazwyczaj nie zmienia się po lematyzacji (zmienia się tylko ich forma), 
# chyba że wprowadzimy dodatkowe filtry.

# Zliczanie
# Zliczanie wystąpień słów (Bag of Words)
word_counts = Counter(lemmatized_tokens)

# Pobranie 10 najpopularniejszych słów
top_10_words = word_counts.most_common(10)

# Rozdzielenie na słowa (oś X) i ich liczebności (oś Y)
words = [item[0] for item in top_10_words]
counts = [item[1] for item in top_10_words]

# Rysowanie wykresu
plt.figure(figsize=(10, 6))
plt.bar(words, counts, color='skyblue')
plt.xlabel('Słowa (Bag of Words)')
plt.ylabel('Liczba wystąpień')
plt.title('10 najczęściej występujących słów w artykule')
plt.xticks(rotation=45) # Obrót etykiet dla lepszej czytelności
plt.tight_layout()
plt.show()

# Generowanie chmury słów
# WordCloud wymaga pojedynczego stringa jako wejścia, więc łączymy z powrotem nasze zlemmatyzowane tokeny
processed_text = " ".join(lemmatized_tokens)

# Tworzenie obiektu WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(processed_text)

# Rysowanie chmury
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') # Wyłączenie osi
plt.title('Chmura Tagów (Word Cloud)')
plt.show()