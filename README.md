# Detekcja Fake News

Zaawansowany projekt uczenia maszynowego do wykrywania fałszywych wiadomości wykorzystujący przetwarzanie języka naturalnego i algorytmy klasyfikacji. Projekt analizuje dataset WELFake, aby zbudować niezawodne modele zdolne do rozróżniania między autentycznymi a spreparowanymi artykułami informacyjnymi.

## Przegląd Projektu

Ten projekt implementuje kompleksowy system detekcji fake news, który:
- Przetwarza i analizuje ponad 72 000 artykułów prasowych
- Wykonuje rozbudowaną eksploracyjną analizę danych (EDA)
- Implementuje liczne algorytmy uczenia maszynowego
- Zapewnia szczegółowe wizualizacje i metryki wydajności
- Identyfikuje kluczowe cechy językowe odróżniające fake news od prawdziwych

## Kluczowe Funkcjonalności

- **Przetwarzanie Danych**: Czyszczenie tekstu, tokenizacja i usuwanie słów-wypełniaczy
- **Eksploracyjna Analiza Danych**: Analiza statystyczna, chmury słów i analiza częstotliwości
- **Inżynieria Cech**: Wektoryzacja TF-IDF z n-gramami
- **Wielokrotne Modele ML**: Regresja Logistyczna, Naive Bayes i Random Forest
- **Kompleksowa Ewaluacja**: Accuracy, precision, recall, F1-score i krzywe ROC
- **Analiza Ważności Cech**: Identyfikacja słów charakteryzujących fake vs prawdziwe news

## Dataset

Kaggle: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/code?datasetId=2093157&sortBy=voteCount 
Projekt wykorzystuje **WELFake Dataset** zawierający:
- **Łączna liczba artykułów**: 72 134
- **Fake News**: ~50% datasetu
- **Prawdziwe News**: ~50% datasetu
- **Cechy**: Tytuł, treść artykułu i etykiety binarne (0=Fake, 1=Prawdziwe)

## Użyte Technologie

### Biblioteki Python
- **Przetwarzanie Danych**: `pandas`, `numpy`
- **Uczenie Maszynowe**: `scikit-learn`
- **Przetwarzanie Języka Naturalnego**: `nltk`
- **Wizualizacja**: `matplotlib`, `seaborn`, `wordcloud`
- **Analiza Tekstu**: `TfidfVectorizer`, `CountVectorizer`

### Algorytmy Uczenia Maszynowego
1. **Regresja Logistyczna** - Klasyfikacja liniowa z regularyzacją
2. **Multinomial Naive Bayes** - Probabilistyczna klasyfikacja tekstu
3. **Random Forest** - Metoda zespołowa z analizą ważności cech

### Najlepszy Model: Random Forest
- **Accuracy**: 96%
- **F1-Score**: 96.2%
- **AUC**: 99.4%

## 📁 Struktura Projektu

```
fakenews/
│
├── fakenews.ipynb          # Główny notebook Jupyter
├── WELFake_Dataset.csv     # Plik z danymi
├── README.md               # Dokumentacja projektu
```

## Jak Zacząć

### Wymagania
```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk
```

### Instalacja
1. Sklonuj repozytorium:
```bash
git clone https://github.com/franeklasinski/Fake-News-Classification.git
cd fakenews
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

3. Pobierz dane NLTK:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

4. Uruchom notebook Jupyter:
```bash
jupyter notebook fakenews.ipynb
```

## Wizualizacje

Projekt zawiera kompleksowe wizualizacje:

1. **Rozkład Klas**: Wykresy kołowe i słupkowe pokazujące rozkład fake vs prawdziwe news
2. **Analiza Tekstu**: Rozkłady długości i liczby słów według typu wiadomości
3. **Chmury Słów**: Wizualna reprezentacja najczęstszych słów w każdej kategorii
4. **Wydajność Modeli**: Krzywe ROC, macierze pomyłek i porównania metryk
5. **Ważność Cech**: Analiza słów najbardziej wskazujących na fake/prawdziwe news

## Metodologia

### Przetwarzanie Danych
1. **Czyszczenie Tekstu**: Usuwanie interpunkcji, konwersja na małe litery
2. **Tokenizacja**: Podział tekstu na pojedyncze słowa
3. **Usuwanie Stopwords**: Filtrowanie popularnych angielskich słów-wypełniaczy
4. **Ekstrakcja Cech**: Wektoryzacja TF-IDF z unigramami i bigramami

### Trenowanie Modeli
1. **Podział Danych**: 80% treningowy, 20% testowy ze stratyfikowanym próbkowaniem
2. **Tworzenie Pipeline**: Połączenie wektoryzacji TF-IDF z klasyfikacją
3. **Ewaluacja Modeli**: Walidacja krzyżowa i kompleksowe metryki
4. **Analiza Cech**: Identyfikacja najważniejszych cech predykcyjnych

## Kontakt

- **GitHub**:(https://github.com/franeklasinski)

## Autor
Franciszek Łasiński
---
