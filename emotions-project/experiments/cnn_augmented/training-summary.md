Klasy: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
Train: 22967 | Val: 5742 | Test: 7178
Augmentacja: TAK
Urzadzenie: cuda
Wagi klas: {'angry': 1.024999976158142, 'disgust': 9.765000343322754, 'fear': 1.0019999742507935, 'happy': 0.5690000057220459, 'neutral': 0.8259999752044678, 'sad': 0.8510000109672546, 'surprise': 1.2799999713897705}

=== Trening ===
Epoka [01/30] | Train loss 1.9137 acc 19.44% | Val loss 1.8406 acc 20.99% | LR 0.00100
  -> nowy best (val acc 20.99%) zapisany
Epoka [02/30] | Train loss 1.8313 acc 26.36% | Val loss 1.7189 acc 31.84% | LR 0.00100
  -> nowy best (val acc 31.84%) zapisany
Epoka [03/30] | Train loss 1.7332 acc 33.60% | Val loss 1.6020 acc 33.32% | LR 0.00100
  -> nowy best (val acc 33.32%) zapisany
Epoka [04/30] | Train loss 1.6401 acc 37.50% | Val loss 1.4966 acc 42.98% | LR 0.00100
  -> nowy best (val acc 42.98%) zapisany
Epoka [05/30] | Train loss 1.6041 acc 39.23% | Val loss 1.4602 acc 42.60% | LR 0.00100
Epoka [06/30] | Train loss 1.5584 acc 40.47% | Val loss 1.4249 acc 42.95% | LR 0.00100
Epoka [07/30] | Train loss 1.5187 acc 42.36% | Val loss 1.4117 acc 43.77% | LR 0.00100
  -> nowy best (val acc 43.77%) zapisany
Epoka [08/30] | Train loss 1.5069 acc 42.56% | Val loss 1.3719 acc 46.99% | LR 0.00100
  -> nowy best (val acc 46.99%) zapisany
Epoka [09/30] | Train loss 1.4844 acc 43.84% | Val loss 1.3188 acc 49.04% | LR 0.00100
  -> nowy best (val acc 49.04%) zapisany
Epoka [10/30] | Train loss 1.4580 acc 44.95% | Val loss 1.3039 acc 49.44% | LR 0.00100
  -> nowy best (val acc 49.44%) zapisany
Epoka [11/30] | Train loss 1.4393 acc 45.43% | Val loss 1.3163 acc 49.43% | LR 0.00100
Epoka [12/30] | Train loss 1.4260 acc 45.84% | Val loss 1.2906 acc 50.03% | LR 0.00100
  -> nowy best (val acc 50.03%) zapisany
Epoka [13/30] | Train loss 1.4125 acc 46.25% | Val loss 1.3110 acc 50.47% | LR 0.00100
  -> nowy best (val acc 50.47%) zapisany
Epoka [14/30] | Train loss 1.4056 acc 46.31% | Val loss 1.3090 acc 47.86% | LR 0.00100
Epoka [15/30] | Train loss 1.3932 acc 46.95% | Val loss 1.2512 acc 52.80% | LR 0.00100
  -> nowy best (val acc 52.80%) zapisany
Epoka [16/30] | Train loss 1.3813 acc 47.72% | Val loss 1.2572 acc 53.88% | LR 0.00100
  -> nowy best (val acc 53.88%) zapisany
Epoka [17/30] | Train loss 1.3744 acc 47.90% | Val loss 1.2646 acc 51.48% | LR 0.00100
Epoka [18/30] | Train loss 1.3670 acc 48.51% | Val loss 1.2502 acc 53.13% | LR 0.00100
Epoka [19/30] | Train loss 1.3614 acc 48.77% | Val loss 1.2542 acc 53.47% | LR 0.00050
Epoka [20/30] | Train loss 1.3271 acc 50.02% | Val loss 1.2327 acc 51.43% | LR 0.00050
Epoka [21/30] | Train loss 1.3078 acc 50.69% | Val loss 1.2330 acc 54.63% | LR 0.00050
  -> nowy best (val acc 54.63%) zapisany
Epoka [22/30] | Train loss 1.3055 acc 50.21% | Val loss 1.2290 acc 55.10% | LR 0.00050
  -> nowy best (val acc 55.10%) zapisany
Epoka [23/30] | Train loss 1.2877 acc 50.44% | Val loss 1.2036 acc 54.32% | LR 0.00050
Epoka [24/30] | Train loss 1.3067 acc 50.58% | Val loss 1.1958 acc 53.90% | LR 0.00050
Epoka [25/30] | Train loss 1.2820 acc 50.96% | Val loss 1.2090 acc 53.45% | LR 0.00025
Epoka [26/30] | Train loss 1.2659 acc 51.96% | Val loss 1.1925 acc 54.96% | LR 0.00025
Epoka [27/30] | Train loss 1.2560 acc 51.79% | Val loss 1.1795 acc 54.58% | LR 0.00025
Early stopping po 27 epokach (brak poprawy przez 5)

Krzywe uczenia: /content/out/learning_curves.png

=== Ewaluacja na zbiorze testowym (najlepszy model) ===
Test accuracy: 55.11%

              precision    recall  f1-score   support

       angry     0.4625    0.4509    0.4567       958
     disgust     0.2653    0.5856    0.3652       111
        fear     0.3604    0.2080    0.2638      1024
       happy     0.7535    0.7993    0.7757      1774
     neutral     0.4770    0.6148    0.5372      1233
         sad     0.4483    0.3512    0.3939      1247
    surprise     0.6583    0.7605    0.7058       831

    accuracy                         0.5511      7178
   macro avg     0.4893    0.5386    0.4997      7178
weighted avg     0.5395    0.5511    0.5383      7178

Confusion matrix: /content/out/confusion_matrix.png

Gotowe. Wszystkie artefakty w: /content/out/

docs/screenshots/cnn_augmented_loss_acc.png

docs/screenshots/cnn_augmented_matrix.png

Test accuracy: 55.11%

              precision    recall  f1-score   support

       angry     0.4625    0.4509    0.4567       958
     disgust     0.2653    0.5856    0.3652       111
        fear     0.3604    0.2080    0.2638      1024
       happy     0.7535    0.7993    0.7757      1774
     neutral     0.4770    0.6148    0.5372      1233
         sad     0.4483    0.3512    0.3939      1247
    surprise     0.6583    0.7605    0.7058       831

    accuracy                         0.5511      7178
   macro avg     0.4893    0.5386    0.4997      7178
weighted avg     0.5395    0.5511    0.5383      7178
