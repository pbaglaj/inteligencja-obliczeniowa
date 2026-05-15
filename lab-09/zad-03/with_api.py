import requests
import pandas as pd
from datetime import datetime

# 1. Konfiguracja zapytania (Brak kluczy API!)
topic = "Netanyahu"
posts_count = 100

print(f"Rozpoczynam pobieranie {posts_count} postów o temacie: '{topic}' z Hacker News...")

# 2. Wywołanie darmowego, otwartego API Algolia dla Hacker News
# Parametr hitsPerPage ustala ile postów pobrać na raz
url = f"https://hn.algolia.com/api/v1/search?query={topic}&hitsPerPage={posts_count}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    posts_data = []

    # 3. Przetwarzanie pobranych danych
    for hit in data.get('hits', []):
        posts_data.append({
            "ID": hit.get('objectID'),
            "Tytuł": hit.get('title'),
            "Autor": hit.get('author'),
            # Konwersja daty z formatu ISO
            "Data": datetime.strptime(hit.get('created_at'), "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S"),
            "Wynik (Punkty)": hit.get('points'),
            "Liczba_komentarzy": hit.get('num_comments'),
            "URL_Postu": f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        })

    # 4. Zapis do pliku CSV
    df = pd.DataFrame(posts_data)
    filename = "pobrane_posty_HackerNews.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"Sukces! Zapisano {len(df)} postów w pliku: {filename}")
    print("\nPodgląd pierwszych 3 postów:")
    print(df[['Tytuł', 'Autor', 'Wynik (Punkty)']].head(3))
else:
    print(f"Błąd pobierania danych. Kod statusu: {response.status_code}")