Klasy: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
Train: 22967 | Val: 5742 | Test: 7178
Augmentacja: NIE
Urzadzenie: cuda
Wagi klas: {'angry': 1.024999976158142, 'disgust': 9.765000343322754, 'fear': 1.0019999742507935, 'happy': 0.5690000057220459, 'neutral': 0.8259999752044678, 'sad': 0.8510000109672546, 'surprise': 1.2799999713897705}

=== Trening ===
Epoka [01/30] | Train loss 1.8299 acc 25.88% | Val loss 1.7178 acc 32.18% | LR 0.00100
  -> nowy best (val acc 32.18%) zapisany
Epoka [02/30] | Train loss 1.6396 acc 37.34% | Val loss 1.5389 acc 42.09% | LR 0.00100
  -> nowy best (val acc 42.09%) zapisany
Epoka [03/30] | Train loss 1.4597 acc 43.49% | Val loss 1.4472 acc 42.51% | LR 0.00100
  -> nowy best (val acc 42.51%) zapisany
Epoka [04/30] | Train loss 1.3247 acc 48.67% | Val loss 1.4122 acc 49.01% | LR 0.00100
  -> nowy best (val acc 49.01%) zapisany
Epoka [05/30] | Train loss 1.2050 acc 52.61% | Val loss 1.3390 acc 51.22% | LR 0.00100
  -> nowy best (val acc 51.22%) zapisany
Epoka [06/30] | Train loss 1.0862 acc 55.99% | Val loss 1.3057 acc 52.02% | LR 0.00100
  -> nowy best (val acc 52.02%) zapisany
Epoka [07/30] | Train loss 0.9754 acc 59.73% | Val loss 1.3404 acc 53.13% | LR 0.00100
  -> nowy best (val acc 53.13%) zapisany
Epoka [08/30] | Train loss 0.9013 acc 62.86% | Val loss 1.3935 acc 54.53% | LR 0.00100
  -> nowy best (val acc 54.53%) zapisany
Epoka [09/30] | Train loss 0.8137 acc 66.07% | Val loss 1.4325 acc 54.93% | LR 0.00100
  -> nowy best (val acc 54.93%) zapisany
Epoka [10/30] | Train loss 0.7371 acc 69.21% | Val loss 1.5564 acc 54.77% | LR 0.00100
Epoka [11/30] | Train loss 0.6567 acc 72.56% | Val loss 1.5425 acc 54.88% | LR 0.00100
Epoka [12/30] | Train loss 0.5927 acc 75.65% | Val loss 1.6196 acc 54.86% | LR 0.00050
Epoka [13/30] | Train loss 0.4643 acc 80.88% | Val loss 1.7873 acc 56.41% | LR 0.00050
  -> nowy best (val acc 56.41%) zapisany
Epoka [14/30] | Train loss 0.4046 acc 83.18% | Val loss 1.9094 acc 55.71% | LR 0.00050
Epoka [15/30] | Train loss 0.3594 acc 85.02% | Val loss 2.0292 acc 56.18% | LR 0.00050
Epoka [16/30] | Train loss 0.3298 acc 86.37% | Val loss 2.0970 acc 55.38% | LR 0.00025
Epoka [17/30] | Train loss 0.2724 acc 88.84% | Val loss 2.2005 acc 56.41% | LR 0.00025
Epoka [18/30] | Train loss 0.2451 acc 90.10% | Val loss 2.3404 acc 55.96% | LR 0.00025
Early stopping po 18 epokach (brak poprawy przez 5)

Krzywe uczenia: /content/out_no_aug/learning_curves.png

=== Ewaluacja na zbiorze testowym (najlepszy model) ===
Test accuracy: 55.67%

              precision    recall  f1-score   support

       angry     0.4525    0.5073    0.4783       958
     disgust     0.6082    0.5315    0.5673       111
        fear     0.4352    0.3740    0.4023      1024
       happy     0.7457    0.7655    0.7555      1774
     neutral     0.4929    0.5036    0.4982      1233
         sad     0.3985    0.3889    0.3937      1247
    surprise     0.7286    0.7268    0.7277       831

    accuracy                         0.5567      7178
   macro avg     0.5517    0.5425    0.5461      7178
weighted avg     0.5544    0.5567    0.5549      7178

Confusion matrix: /content/out_no_aug/confusion_matrix.png

Gotowe. Wszystkie artefakty w: /content/out_no_aug/