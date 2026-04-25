Alfa (Wpływ feromonu): Określa, jak bardzo mrówki "ufają" śladom zapachowym pozostawionym przez poprzedniczki.

Za wysoka alfa powoduje zjawisko przedwczesnej zbieżności – wszystkie mrówki szybko wchodzą na jedną ścieżkę (niekoniecznie najlepszą) i utykają w tzw. ekstremum lokalnym.

Beta (Wpływ widoczności/odległości): Jest to heurystyka (wiedza o problemie). Im wyższa Beta, tym mrówka chętniej idzie do najbliższego fizycznie miasta, ignorując to, że gdzie indziej jest dużo feromonu.

Za wysoka beta sprawia, że algorytm zamienia się w zwykły algorytm zachłanny ("idź do najbliższego sąsiada"), tracąc zdolność całej roju do znajdowania sprytnych, długoterminowych skrótów.

Evaporation Rate (Współczynnik parowania): Określa, ile procent feromonu znika po każdej iteracji.

Szybkie parowanie (np. 0.8) zmusza mrówki do ciągłego poszukiwania nowych ścieżek (eksploracja), ponieważ stare ślady szybko znikają.

Wolne parowanie (np. 0.1) sprawia, że ścieżki utrwalają się na długo (eksploatacja), co może przyspieszyć znalezienie wyniku, ale zwiększa ryzyko ominięcia lepszej trasy.


## d)
Dla człowieka wymyślenie dobrej trasy na gridzie jest trywialne. Nasz mózg uwielbia wzory, więc naturalnie zastosujemy tzw. wzór kosiarki (lub grzebienia):

Zaczynamy w (0,0) i idziemy w górę do (0,40) (odległość: 40).

Przeskakujemy w prawo na (10,40) (odległość: 10).

Schodzimy w dół do (10,10) (odległość: 30).

Przeskakujemy w prawo na (20,10) i idziemy w górę do (20,40) (odległość: 10 + 30 = 40)...

Powtarzamy ten zygzak, aż dotrzemy do (40,40).

Z punktu (40,40) musimy wrócić do startu zaliczając dolny rząd: idziemy w dół do (40,0), w lewo do (10,0) i zamykamy cykl w (0,0).

Długość takiej "ludzkiej" trasy:
Taki zygzakowaty wzór składa się z samych linii prostych. Jeśli go dokładnie zsumujemy, długość wyniesie dokładnie 280.

Choć trasa 280 wydaje się idealna, nie jest najkrótsza! Szukanie idealnego optimum na siatce o nieparzystej liczbie punktów (u nas 5x5 = 25) skrywa piękny matematyczny paradoks.Wyobraź sobie tę siatkę jako szachownicę. Każdy ruch w pionie lub poziomie zmienia kolor pola (z czarnego na białe). Skoro mamy 25 pól, oznacza to, że mamy np. 13 czarnych i 12 białych. Niemożliwe jest przejście przez wszystkie pola i powrót do startu używając TYLKO ruchów góra/dół/lewo/prawo, ponieważ wymagałoby to równej liczby pól obu kolorów!Musimy wykonać dokładnie jeden krok po skosie, aby zbalansować parzystość.Krok prosty ma długość: 10Najkrótszy krok po skosie ma długość: $\sqrt{10^2 + 10^2} \approx 14.14$Idealna trasa polega na przejściu 24 odcinków prostych i 1 ukośnego.Zatem absolutne minimum to: $(24 \cdot 10) + 14.14 = 254.14$.

Algorytmy heurystyczne nienawidzą gridów. Przez ogromną symetrię istnieje mnóstwo tras o bardzo zbliżonej długości (tzw. ekstrema lokalne). Feromony łatwo rozkładają się na symetrycznych ścieżkach, co dezorientuje mrówki.