import numpy as np

def sig(x):
    return 1 / (1 + np.exp(-x))

def train_one_step():
    x1 = 0.6
    x2 = 0.1
    y = 0.8         # Target (oczekiwana wartość wyjściowa)
    eta = 0.1       # Szybkość uczenia (Learning Rate)

    # Wagi początkowe
    w1 = 0.2;  w2 = -0.3
    w3 = -0.5; w4 = 0.1
    w5 = 0.3;  w6 = -0.4
    
    # Biasy początkowe
    b1 = 0.4;  b2 = -0.2;  b3 = 0.2

    print("STAN PRZED UCZENIEM")
    
    # 2. FORWARD PROPAGATION
    z1 = w1 * x1 + w2 * x2 + b1
    z2 = w3 * x1 + w4 * x2 + b2

    h1 = sig(z1)
    h2 = sig(z2)

    # Wyjście sieci
    y_hat = w5 * h1 + w6 * h2 + b3
    
    # Błąd średniokwadratowy (MSE dla jednej próbki: E = (y_hat - y)^2)
    loss = (y_hat - y) ** 2
    print(f"Predykcja sieci: {y_hat:.4f} | Target: {y}")
    print(f"Początkowy błąd (MSE): {loss:.4f}")
    print(f"Początkowa waga w5: {w5:.4f}")

    # 3. BACKWARD PROPAGATION
    # KROK A: Pochodna błędu względem wyniku wyjściowego
    # Skoro E = (y_hat - y)^2, to pochodna dE/dy_hat = 2 * (y_hat - y)
    dE_dyhat = 2 * (y_hat - y)

    # KROK B: Gradienty dla warstwy wyjściowej (w5, w6, b3)
    # dE/dw5 = dE/dy_hat * dy_hat/dw5 (pochodna y_hat względem w5 to po prostu h1)
    dE_dw5 = dE_dyhat * h1
    dE_dw6 = dE_dyhat * h2
    dE_db3 = dE_dyhat * 1

    # KROK C: Gradienty dla warstwy ukrytej
    # Najpierw liczymy, jak bardzo wyjście warstwy ukrytej (h1, h2) wpłynęło na błąd
    dE_dh1 = dE_dyhat * w5
    dE_dh2 = dE_dyhat * w6

    # Przechodzimy wstecz przez funkcję aktywacji. 
    # Pochodna sigmoidy to: sig(x) * (1 - sig(x)), czyli u nas: h * (1 - h)
    dE_dz1 = dE_dh1 * (h1 * (1 - h1))
    dE_dz2 = dE_dh2 * (h2 * (1 - h2))

    # Wyliczamy ostateczne gradienty dla wag wejściowych (w1, w2, w3, w4 i biasów)
    dE_dw1 = dE_dz1 * x1
    dE_dw2 = dE_dz1 * x2
    dE_db1 = dE_dz1 * 1

    dE_dw3 = dE_dz2 * x1
    dE_dw4 = dE_dz2 * x2
    dE_db2 = dE_dz2 * 1

    # 4. GRADIENT DESCENT (Spadek gradientu)
    # Aktualizacja wag i biasów według wzoru: W_nowe = W_stare - eta * gradient
    w1_new = w1 - eta * dE_dw1
    w2_new = w2 - eta * dE_dw2
    b1_new = b1 - eta * dE_db1

    w3_new = w3 - eta * dE_dw3
    w4_new = w4 - eta * dE_dw4
    b2_new = b2 - eta * dE_db2

    w5_new = w5 - eta * dE_dw5
    w6_new = w6 - eta * dE_dw6
    b3_new = b3 - eta * dE_db3

    print("\n--- STAN PO 1 KROKU UCZENIA ---")
    print(f"Nowa waga w5: {w5_new:.4f} (zmieniona z {w5:.4f})")
    
    # SPRAWDZENIE: Czy po poprawie wag błąd jest mniejszy?
    z1_new = w1_new * x1 + w2_new * x2 + b1_new
    z2_new = w3_new * x1 + w4_new * x2 + b2_new
    
    h1_new = sig(z1_new)
    h2_new = sig(z2_new)
    
    y_hat_new = w5_new * h1_new + w6_new * h2_new + b3_new
    loss_new = (y_hat_new - y) ** 2
    
    print(f"Nowa predykcja sieci: {y_hat_new:.4f}")
    print(f"Nowy błąd (MSE): {loss_new:.4f}")

train_one_step()