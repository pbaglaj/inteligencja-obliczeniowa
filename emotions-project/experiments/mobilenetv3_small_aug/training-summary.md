Klasy (folder -> nazwa): {'1': 'Surprise', '2': 'Fear', '3': 'Disgust', '4': 'Happiness', '5': 'Sadness', '6': 'Anger', '7': 'Neutral'}
Train: 10430 | Val: 1841 | Test: 3068
Augmentacja: TAK
Model: MobileNetV3-small (pretrained ImageNet)
Urzadzenie: cuda
Downloading: "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
100% 9.83M/9.83M [00:00<00:00, 63.4MB/s]
Wagi klas: {'Surprise': 1.3609999418258667, 'Fear': 6.478000164031982, 'Disgust': 2.443000078201294, 'Happiness': 0.36500000953674316, 'Sadness': 0.8899999856948853, 'Anger': 2.492000102996826, 'Neutral': 0.6949999928474426}

=== Faza 1: warm-up (3 epok, zamrozony backbone) ===
Epoka [01/20] (warmup) | Train loss 1.6559 acc 37.23% | Val loss 1.5691 acc 39.11% | LR 0.001000
  -> nowy best (val acc 39.11%) zapisany
Epoka [02/20] (warmup) | Train loss 1.5294 acc 44.92% | Val loss 1.5362 acc 36.72% | LR 0.001000
Epoka [03/20] (warmup) | Train loss 1.4525 acc 47.16% | Val loss 1.4494 acc 47.15% | LR 0.001000
  -> nowy best (val acc 47.15%) zapisany

=== Faza 2: fine-tuning (odmrozony caly model, LR=0.0001) ===
Epoka [04/20] (finetune) | Train loss 1.2367 acc 57.03% | Val loss 1.2167 acc 57.52% | LR 0.000099
  -> nowy best (val acc 57.52%) zapisany
Epoka [05/20] (finetune) | Train loss 1.0594 acc 63.30% | Val loss 1.1258 acc 60.78% | LR 0.000097
  -> nowy best (val acc 60.78%) zapisany
Epoka [06/20] (finetune) | Train loss 0.9517 acc 67.71% | Val loss 1.0859 acc 66.38% | LR 0.000093
  -> nowy best (val acc 66.38%) zapisany
Epoka [07/20] (finetune) | Train loss 0.8615 acc 70.78% | Val loss 1.0332 acc 68.06% | LR 0.000087
  -> nowy best (val acc 68.06%) zapisany
Epoka [08/20] (finetune) | Train loss 0.7863 acc 73.04% | Val loss 0.9870 acc 69.64% | LR 0.000080
  -> nowy best (val acc 69.64%) zapisany
Epoka [09/20] (finetune) | Train loss 0.7260 acc 74.46% | Val loss 1.0015 acc 71.16% | LR 0.000073
  -> nowy best (val acc 71.16%) zapisany
Epoka [10/20] (finetune) | Train loss 0.6817 acc 75.87% | Val loss 0.9610 acc 70.89% | LR 0.000064
Epoka [11/20] (finetune) | Train loss 0.6197 acc 77.02% | Val loss 0.9793 acc 72.13% | LR 0.000055
  -> nowy best (val acc 72.13%) zapisany
Epoka [12/20] (finetune) | Train loss 0.5863 acc 78.52% | Val loss 0.9878 acc 71.97% | LR 0.000046
Epoka [13/20] (finetune) | Train loss 0.5537 acc 79.42% | Val loss 0.9796 acc 72.95% | LR 0.000037
  -> nowy best (val acc 72.95%) zapisany
Epoka [14/20] (finetune) | Train loss 0.5268 acc 80.34% | Val loss 0.9662 acc 73.60% | LR 0.000028
  -> nowy best (val acc 73.60%) zapisany
Epoka [15/20] (finetune) | Train loss 0.4952 acc 81.30% | Val loss 0.9743 acc 73.93% | LR 0.000021
  -> nowy best (val acc 73.93%) zapisany
Epoka [16/20] (finetune) | Train loss 0.4841 acc 81.44% | Val loss 0.9638 acc 73.38% | LR 0.000014
Epoka [17/20] (finetune) | Train loss 0.4696 acc 81.74% | Val loss 0.9817 acc 73.49% | LR 0.000008
Epoka [18/20] (finetune) | Train loss 0.4620 acc 81.89% | Val loss 0.9883 acc 73.93% | LR 0.000004
Epoka [19/20] (finetune) | Train loss 0.4481 acc 82.41% | Val loss 0.9807 acc 73.76% | LR 0.000002
Epoka [20/20] (finetune) | Train loss 0.4492 acc 82.65% | Val loss 0.9839 acc 73.76% | LR 0.000001
Early stopping po 20 epokach (brak poprawy przez 5)

Krzywe uczenia: /content/out_small/learning_curves.png

=== Ewaluacja na zbiorze testowym (najlepszy model) ===
Test accuracy: 75.00%

              precision    recall  f1-score   support

    Surprise     0.7268    0.7842    0.7544       329
        Fear     0.4231    0.5946    0.4944        74
     Disgust     0.4101    0.4562    0.4320       160
   Happiness     0.9060    0.8540    0.8792      1185
     Sadness     0.7681    0.6653    0.7130       478
       Anger     0.5430    0.7407    0.6266       162
     Neutral     0.7010    0.7000    0.7005       680

    accuracy                         0.7500      3068
   macro avg     0.6397    0.6850    0.6572      3068
weighted avg     0.7632    0.7500    0.7544      3068

Confusion matrix: /content/out_small/confusion_matrix.png

Gotowe. Wszystkie artefakty w: /content/out_small/