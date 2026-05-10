# Demand Forecast Lab

Aplikacja webowa do **porównywania algorytmów prognozowania kwantylowego** (P50 / P90) dla zmiennej **OrdersIn** z horyzontem **6 miesięcy**, rolling backtest i wspólnym zbiorem testowym.

## Stack

- **Frontend:** Next.js 14 (React), Tailwind CSS  
- **Backend:** Python 3.10–3.12 (zalecane), FastAPI  
- **ML:** pandas, NumPy, scikit-learn (Gradient Boosting Quantile), CatBoost, LightGBM  

> **Uwaga:** Python **3.14** i starsze środowiska bez kół dla `pandas` mogą wymagać innej wersji Pythona lub instalacji ze skompilowanymi wheelami.

## Struktura katalogów

```
Aplikacja_4/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI, endpointy
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── session_store.py
│   │   └── services/
│   │       ├── preprocessing.py
│   │       ├── feature_engineering.py
│   │       ├── model_training.py
│   │       ├── backtest.py
│   │       ├── metrics.py
│   │       ├── ranking.py
│   │       └── comparison_runner.py
│   ├── data/sample_monthly.csv     # Przykładowe dane (generator)
│   ├── generate_sample_data.py
│   └── requirements.txt
├── frontend/                       # Next.js dashboard
└── README.md
```

## Backend — instalacja i uruchomienie

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

### Endpointy API

| Metoda | Ścieżka | Opis |
|--------|---------|------|
| POST | `/load-afg-bundled` | Ładuje z dysku i scala **oba** CSV AFG z katalogu projektu |
| GET | `/preview-data?session_id=` | Podgląd danych |
| POST | `/run-comparison` | Start joba (tło): trening + backtest |
| GET | `/run-progress?session_id=` | Postęp (0–1) |
| GET | `/model-results?session_id=` | Wyniki (`completed` / `running` / `error`) |
| GET | `/export-results?session_id=&fmt=csv|xlsx` | Eksport tabeli metryk |

Sesje i pliki tymczasowe: katalog `backend/data/sessions/` (tworzony automatycznie).

### Pliki AFG (feature-only + train)

W katalogu głównym projektu (`Aplikacja_4`) umieść:

- `AFG_ML_FEATUREONLY_SKU_EOM_STRICT.csv`
- `AFG_ML_TRAIN_SKU_EOM_STRICT.csv`

Łączenie: **`POST /load-afg-bundled`** (lub przycisk w UI) scala po `ProductID` + `EOM`; dla wspólnych kolumn preferowana jest wartość z pliku **train**; kolumny tylko w train (np. `KPI_LostSales_*`, `KPI_FillRate_*`) trafiają do zbioru. Dodawany jest alias **`OrdersIn`** z `KPI_OrdersIn_Qty`.

Inny katalog z CSV: zmienna środowiskowa `AFG_PROJECT_ROOT` (ścieżka bezwzględna lub względna przy starcie backendu).

### Przykładowe dane

```powershell
cd backend
py generate_sample_data.py --months 48 --skus 3 --out data/sample_monthly.csv
```

## Frontend

Domyślnie UI woła API przez **proxy Next.js** (`/api-proxy` → backend), więc **nie potrzebujesz ustawiać CORS** ani `NEXT_PUBLIC_API_URL`, o ile backend działa na **http://127.0.0.1:8001**.

```powershell
cd frontend
npm install
npm run dev
```

Inny adres backendu (serwer Next proxy’uje żądania):

```powershell
$env:API_PROXY_TARGET="http://127.0.0.1:9000"
npm run dev
```

Bezpośrednie wywołania API z przeglądarki (wtedy backend musi mieć CORS):

```powershell
$env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8001"
npm run dev
```

Otwórz [http://localhost:3000](http://localhost:3000) (lub port wskazany przez Next).

## Modelowanie i metryki

- **Modele:** CatBoost Quantile, LightGBM Quantile, sklearn `GradientBoostingRegressor(loss='quantile')`, plus **baseline** (sezonowa naiwność + konserwatywne P90).  
- **Cechy:** lagi 1–3, 6, 12; średnie i odchylenia rolki 3/6/12; miesiąc, kwartał, rok; trend; `sin/cos` sezonowości; identyfikatory SKU / lokalizacji jako cechy kategoryczne.  
- **Kalibracja P90:** prosty mnożnik dopasowujący pokrycie do ~90% na punktach backtestu (per model).  
- **Metryki serwisowe** (LostSales / FillRate): uproszczony proxy przy obecności kolumny zrealizowanej ilości; w przeciwnym razie `N/A` z wyjaśnieniem w UI.

## Licencja

Projekt przykładowy — dostosuj i rozszerz pod własne wdrożenie.
