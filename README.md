# Analiza i segmentacja klientów
## Opis repozytorium
W pliku `eda.ipynb` znajduje się eksploracyjna analiza danych, a w pliku `segmentation.ipynb` segmentacja klientów. Folder `src` zawiera kod źródłowy wykorzystywany w obu notebookach.

## Uruchomienie kodu
By odtworzyć notebooki, należy:
1. Sklonować to repozytorium
```
git clone https://github.com/zuzg/customer-analysis.git
```
2. W repozytorium utworzyć folder `data` i umieścić tam rozpakowany plik z danymi.
3. Utworzyć środowisko. Przygotowany plik jest kompatybilny z menadżerem pakietów conda. Zmienić folder na repozytorium i wykonać komendę
```
conda env create -f environment.yml
```