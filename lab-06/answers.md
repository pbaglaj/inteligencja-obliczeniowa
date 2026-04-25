Czy w zbiorze występują koty przypominające psy albo psy przypominające koty? Tak. W zbiorze walidacyjnym mogą pojawić się obrazy graniczne, czyli takie, które wizualnie są trudne do jednoznacznego przypisania do klasy kota albo psa.

Ile takich przypadków można wskazać? Tyle, ile model faktycznie błędnie sklasyfikuje podczas uruchomienia. Skrypt po treningu wypisuje liczbę pomyłek oraz rozbicie na dwie grupy: kot -> pies i pies -> kot.

Czy potrafisz znaleźć i pokazać konkretne obrazy błędnie sklasyfikowane przez model? Tak. Skrypt zbiera błędne predykcje, sortuje je według pewności i pokazuje najciekawsze przypadki na wykresie.

Podsumowanie: sekcja d) jest obsłużona poprawnie. Program pokazuje krzywe uczenia dla zbioru treningowego i walidacyjnego, a po treningu analizuje i prezentuje błędne klasyfikacje.

Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=16384, out_features=512, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=2, bias=True)
  )
)
Rozpoczynam trening...
Epoka 1/3 (199.7s) -> Train Loss: 0.6714, Acc: 0.5853 | Val Loss: 0.6485, Acc: 0.6220
Epoka 2/3 (187.0s) -> Train Loss: 0.5964, Acc: 0.6788 | Val Loss: 0.5885, Acc: 0.6912
Epoka 3/3 (196.1s) -> Train Loss: 0.5354, Acc: 0.7319 | Val Loss: 0.5179, Acc: 0.7416

Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to C:\Users\Admin/.cache\torch\hub\checkpoints\resnet18-f37072fd.pth
100.0%
Model ResNet gotowy do szybkiego dotrenowania!



dane wczytane! zbiór treningowy: 20000 zdjęć. zbiór walidacyjny: 5000 zdjęć.
Model został zbudowany i wysłany na: cpu
SimpleCNN(
  (conv_layers): Sequential(
    (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU()
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc_layers): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=16384, out_features=512, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=2, bias=True)
  )
)
Brak zapisanego modelu, rozpoczynam trening od zera...
Rozpoczynam trening...
Epoka 1/3 (149.2s) -> Train Loss: 0.6461, Acc: 0.6183 | Val Loss: 0.5699, Acc: 0.7114
Epoka 2/3 (191.5s) -> Train Loss: 0.5612, Acc: 0.7111 | Val Loss: 0.5310, Acc: 0.7334
Epoka 3/3 (186.3s) -> Train Loss: 0.5126, Acc: 0.7470 | Val Loss: 0.5115, Acc: 0.7452
Model został zapisany do pliku cat_dog_model.pth
Model ResNet gotowy do szybkiego dotrenowania!