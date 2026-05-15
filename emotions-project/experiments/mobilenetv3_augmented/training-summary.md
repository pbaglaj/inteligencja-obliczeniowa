Klasy (folder -> nazwa): {'1': 'Surprise', '2': 'Fear', '3': 'Disgust', '4': 'Happiness', '5': 'Sadness', '6': 'Anger', '7': 'Neutral'}
Train: 10430 | Val: 1841 | Test: 3068
Augmentacja: TAK
Model: MobileNetV3-large (pretrained ImageNet)
Urzadzenie: cuda
Downloading: "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth" to /root/.cache/torch/hub/checkpoints/mobilenet_v3_large-5c1a4163.pth
100% 21.1M/21.1M [00:00<00:00, 146MB/s]
Wagi klas: {'Surprise': 1.3609999418258667, 'Fear': 6.478000164031982, 'Disgust': 2.443000078201294, 'Happiness': 0.36500000953674316, 'Sadness': 0.8899999856948853, 'Anger': 2.492000102996826, 'Neutral': 0.6949999928474426}

=== Faza 1: warm-up (3 epok, zamrozony backbone) ===
Epoka [01/20] (warmup) | Train loss 1.6843 acc 34.42% | Val loss 1.6560 acc 45.14% | LR 0.001000
  -> nowy best (val acc 45.14%) zapisany
Epoka [02/20] (warmup) | Train loss 1.5625 acc 42.14% | Val loss 1.5590 acc 45.46% | LR 0.001000
  -> nowy best (val acc 45.46%) zapisany
Epoka [03/20] (warmup) | Train loss 1.5108 acc 44.69% | Val loss 1.5139 acc 41.28% | LR 0.001000

=== Faza 2: fine-tuning (odmrozony caly model, LR=0.0001) ===
Epoka [04/20] (finetune) | Train loss 1.2565 acc 54.64% | Val loss 1.2166 acc 58.17% | LR 0.000099
  -> nowy best (val acc 58.17%) zapisany
Epoka [05/20] (finetune) | Train loss 0.9831 acc 65.33% | Val loss 1.1015 acc 60.84% | LR 0.000097
  -> nowy best (val acc 60.84%) zapisany
Epoka [06/20] (finetune) | Train loss 0.8163 acc 70.83% | Val loss 1.0250 acc 63.28% | LR 0.000093
  -> nowy best (val acc 63.28%) zapisany
Epoka [07/20] (finetune) | Train loss 0.6846 acc 74.87% | Val loss 1.0677 acc 69.53% | LR 0.000087
  -> nowy best (val acc 69.53%) zapisany
Epoka [08/20] (finetune) | Train loss 0.5545 acc 78.55% | Val loss 1.0217 acc 70.56% | LR 0.000080
  -> nowy best (val acc 70.56%) zapisany
Epoka [09/20] (finetune) | Train loss 0.4732 acc 81.59% | Val loss 0.9924 acc 73.66% | LR 0.000073
  -> nowy best (val acc 73.66%) zapisany
Epoka [10/20] (finetune) | Train loss 0.3966 acc 83.77% | Val loss 1.0514 acc 73.55% | LR 0.000064
Epoka [11/20] (finetune) | Train loss 0.3358 acc 86.05% | Val loss 1.0743 acc 73.98% | LR 0.000055
  -> nowy best (val acc 73.98%) zapisany
Epoka [12/20] (finetune) | Train loss 0.2857 acc 87.51% | Val loss 1.0010 acc 75.29% | LR 0.000046
  -> nowy best (val acc 75.29%) zapisany
Epoka [13/20] (finetune) | Train loss 0.2605 acc 88.72% | Val loss 1.1265 acc 74.85% | LR 0.000037
Epoka [14/20] (finetune) | Train loss 0.2156 acc 90.39% | Val loss 1.1839 acc 74.80% | LR 0.000028
Epoka [15/20] (finetune) | Train loss 0.2031 acc 91.02% | Val loss 1.1753 acc 75.94% | LR 0.000021
  -> nowy best (val acc 75.94%) zapisany
Epoka [16/20] (finetune) | Train loss 0.1845 acc 91.71% | Val loss 1.2530 acc 76.37% | LR 0.000014
  -> nowy best (val acc 76.37%) zapisany
Epoka [17/20] (finetune) | Train loss 0.1793 acc 91.67% | Val loss 1.2919 acc 76.75% | LR 0.000008
  -> nowy best (val acc 76.75%) zapisany
Epoka [18/20] (finetune) | Train loss 0.1575 acc 92.86% | Val loss 1.2788 acc 77.08% | LR 0.000004
  -> nowy best (val acc 77.08%) zapisany
Epoka [19/20] (finetune) | Train loss 0.1638 acc 92.65% | Val loss 1.2962 acc 76.97% | LR 0.000002
Epoka [20/20] (finetune) | Train loss 0.1590 acc 92.93% | Val loss 1.3021 acc 76.91% | LR 0.000001

Krzywe uczenia: /content/out/learning_curves.png

=== Ewaluacja na zbiorze testowym (najlepszy model) ===
Test accuracy: 78.10%

              precision    recall  f1-score   support

    Surprise     0.7597    0.8359    0.7959       329
        Fear     0.6230    0.5135    0.5630        74
     Disgust     0.4885    0.5312    0.5090       160
   Happiness     0.9179    0.8684    0.8925      1185
     Sadness     0.7461    0.7071    0.7261       478
       Anger     0.6723    0.7346    0.7021       162
     Neutral     0.7111    0.7529    0.7314       680

    accuracy                         0.7810      3068
   macro avg     0.7027    0.7062    0.7028      3068
weighted avg     0.7859    0.7810    0.7825      3068

Confusion matrix: /content/out/confusion_matrix.png

Gotowe. Wszystkie artefakty w: /content/out/