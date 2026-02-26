import math
import datetime

imie = input("Podaj swoje imię: ")
rok_urodzenia = int(input("Podaj swój rok urodzenia: "))
miesiac_urodzenia = int(input("Podaj swój miesiąc urodzenia: "))
dzien_urodzenia = int(input("Podaj swój dzień urodzenia: "))

print(f"Cześć {imie}!")
dni_zycia = datetime.date.today() - datetime.date(rok_urodzenia, miesiac_urodzenia, dzien_urodzenia)
dni_zycia = dni_zycia.days
print(f"Przeżyłeś już około {dni_zycia} dni")

fizyczna_fala = math.sin(2 * math.pi * dni_zycia / 23)
emocjonalna_fala = math.sin(2 * math.pi * dni_zycia / 28)
intelektualna_fala = math.sin(2 * math.pi * dni_zycia / 33)

print(f"Twoja fizyczna fala: {fizyczna_fala:.2f}")
print(f"Twoja emocjonalna fala: {emocjonalna_fala:.2f}")
print(f"Twoja intelektualna fala: {intelektualna_fala:.2f}")

jutrzejsza_fizyczna_fala = math.sin(2 * math.pi * (dni_zycia + 1) / 23)
jutrzejsza_emocjonalna_fala = math.sin(2 * math.pi * (dni_zycia + 1) / 28)
jutrzejsza_intelektualna_fala = math.sin(2 * math.pi * (dni_zycia + 1) / 33)

if ((fizyczna_fala + emocjonalna_fala + intelektualna_fala) / 3 > 0.5):
    print("Gratuluje dobrego wyniku!")
else:
    print("Nie martw się!")
    if ((jutrzejsza_fizyczna_fala + jutrzejsza_emocjonalna_fala + jutrzejsza_intelektualna_fala) / 3 > 0.5):
        print("Jutro będzie lepszy dzień!")

# Czas wykonania: 15 minut

