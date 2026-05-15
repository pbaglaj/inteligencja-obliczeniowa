## Klasyfikacja (obrazy): Rozpoznawanie emocji na twarzy w czasie rzeczywistym

MediaPipe Face Landmarker (Następca Face Mesh)
To obecnie najpotężniejsze narzędzie od Google do pracy lokalnej (on-device). Nie daje ono bezpośrednio etykiety "Happy", ale dostarcza tzw. Face Blendshapes.

Jak to działa: Model mapuje twarz na 478 punktów i zwraca 52 współczynniki (blendshapes), które odpowiadają za konkretne ruchy mięśni (np. jawOpen, mouthSmileLeft, eyeBlinkLeft).

Zastosowanie: Na podstawie tych wartości możesz matematycznie wyliczyć emocje (np. jeśli mouthSmile > 0.7, to mamy uśmiech).

Zaleta: Działa w czasie rzeczywistym na przeglądarce i telefonie.