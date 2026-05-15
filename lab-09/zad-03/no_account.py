import requests
import pandas as pd
import time

topic = "Artificial Intelligence"
print(f"Rozpoczynam pobieranie postów z Reddita (bez konta) o temacie: '{topic}'...")

# Musimy ustawić unikalny User-Agent, inaczej Reddit odrzuci zapytanie (błąd 429)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Szukamy postów w całym serwisie
url = f"https://www.reddit.com/search.json?q={topic}&limit=100"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    posts_data = []
    
    # Przechodzimy przez strukturę JSON Reddita
    children = data.get('data', {}).get('children', [])
    
    for post in children:
        post_info = post['data']
        posts_data.append({
            "Tytuł": post_info.get('title'),
            "Autor": post_info.get('author'),
            "Subreddit": post_info.get('subreddit_name_prefixed'),
            "Wynik (Upvotes)": post_info.get('score'),
            "Liczba_komentarzy": post_info.get('num_comments'),
            "Tekst": post_info.get('selftext', '')[:500] # Ograniczenie tekstu do 500 znaków
        })
        
    df = pd.DataFrame(posts_data)
    filename = "pobrane_posty_Reddit_NoAccount.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Sukces! Zapisano {len(df)} postów do pliku {filename}.")
else:
    print(f"Błąd! Reddit zablokował dostęp. Kod: {response.status_code}")