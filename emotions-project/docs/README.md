# Datasety

Folder ignorowany w repo (`.gitignore`). Datasety pobierz lokalnie:

## FER2013

7 klas emocji (`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`),
obrazy 48x48 grayscale, ~28k train / ~7k test.

```
data/FER2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── ...
└── test/
    ├── angry/
    ├── ...
```

Zrodlo: <https://www.kaggle.com/datasets/msambare/fer2013>

## RAF-DB

7 klas emocji (`Surprise`, `Fear`, `Disgust`, `Happiness`, `Sadness`, `Anger`, `Neutral`),
obrazy 100x100 RGB aligned, ~12k train / ~3k test. Foldery numerowane 1..7.

```
data/RAF-DB/
├── DATASET/
│   ├── train/
│   │   ├── 1/   # Surprise
│   │   ├── 2/   # Fear
│   │   ├── ...
│   └── test/
│       ├── 1/
│       ├── ...
├── train_labels.csv
└── test_labels.csv
```

Zrodlo: <http://www.whdeng.cn/RAF/model1.html>
