# MSCapital — Real Financial Market Forecasting

Piyasa mikroyapısı verisinden kısa vadeli getiri tahmini yapan, uçtan uca
production'a yakın bir ML sistemi. Kaggle
[MSCapital](https://www.kaggle.com/competitions/ms-capital-real-financial-market-forecasting)
yarışması verisi üzerine kurulu.

> **Araştırma amaçlıdır. Yatırım tavsiyesi değildir.**
> Backtest modülü stratejinin karlılığını değil, modelin sıralama gücünü ölçmek içindir.

---

## Veri: ölçülen gerçekler

Hiçbiri varsayım değil — satır sayıları Arrow footer'ından, dağılımlar veriden okundu.

| Dosya | Satır | Kolon | Disk | Açılmış RAM |
|---|---:|---:|---:|---:|
| `train/market.feather` | 221,756,611 | 13 | 4.10 GiB | **11.53 GB** |
| `train/order.feather` | 170,056,583 | 6 | 1.21 GiB | 3.06 GB |
| `train/transaction.feather` | 103,970,264 | 5 | 476 MiB | 1.77 GB |
| `train/label.feather` | 1,257,637 | 3 | 9.6 MiB | — |
| `test/*` (3 dosya) | 308,733,861 | — | 3.47 GiB | 9.51 GB |
| **Toplam** | **804,517,319** | | **9.26 GiB** | **25.9 GB** |

**Yapı.** Her `sample_id` bağımsız, anonim bir gözlem penceresidir. Sembol/enstrüman
kolonu **yoktur** — bu yüzden sample'lar arası tarihsel state üretilemez ve problem
1,257,637 satırlık tabular regresyona indirgenir.

**Pencere uzunlukları tabloya göre farklıdır** (ölçüldü):

| Tablo | Pencere | Sample başına satır | Not |
|---|---:|---:|---|
| market | **600 sn** | 176.3 | ~3.4 sn'de bir snapshot, satır tavanı yok (max 212) |
| order | 60 sn | 135.2 | **999 satırda tavan** → kırpma feature'ı |
| transaction | 60 sn | 82.7 | **999 satırda tavan** → kırpma feature'ı |

`seconds_before_predict` tahmin anına geriye uzaklıktır ve sample içinde azalan sıradadır;
`0` tahmin anına en yakın tick. Değer her zaman `>= 0` olduğu için **look-ahead yapısal
olarak imkânsızdır**.

**Kodlamalar ampirik olarak çözüldü** (fiyatlar mid ≈ 1.0'a normalize olduğu için):

| Kod | Anlam | Kanıt |
|---|---|---|
| `side = 0` | BID | ort. fiyat 0.9979 (mid'in altında) |
| `side = 1` | ASK | ort. fiyat 1.0036 (mid'in üstünde) |
| `order_action = 0` | NEW | 128.1M olay |
| `order_action = 1` | CANCEL | 42.0M olay; NEW ≈ CANCEL + TRANSACTION dengesi tutuyor |

**`price = 0` bir fiyat değil, "bu seviye boş" sentinel'idir** — her zaman `volume = 0`
ile birlikte gelir. Gerçek fiyatlar 0.909–1.052 aralığında. Temizlenmezse `rel_spread`
ortalaması −0.0064 çıkar (doğrusu +0.001264). Gerçek çapraz defter **yoktur** (0 satır).

**Hedef.** std 0.002618 (26 bps), medyan tam 0 (%5.54 tam-sıfır — tick-size etkisi),
sample'lar arası otokorelasyon ≈ 0. Aylık std 2.69× oynuyor → rejim kayması.

---

## Mimari

```
Kaggle feather (tek record batch, 11.5 GB açılmış)
        │  kolon-grubu dönüştürücü (tepe RAM 7.9 GB)
        ▼
   Parquet parçaları ──► BigQuery
        │                 mscapital_raw → staging → features → mart
        │                 GROUP BY sample_id: 804M satır → 1.26M satır
        ▼
 dataset_train.parquet (1.39 GB, 294 feature)
        │
        ▼
 Walk-forward CV ──► MLflow ──► Model Registry ──► FastAPI ──► Streamlit
```

### Neden kolon-grubu dönüştürücü

Yarışma dosyalarının her biri **tek bir Arrow record batch** tutar. Sonuç: satır bazlı
streaming imkânsız, `memory_map` faydasız (buffer'lar sıkıştırılmış), ve 16 GB RAM'de
`market` tek seferde okunamaz.

Çözüm: Arrow IPC her buffer'ı ayrı sıkıştırır ve `read_table(columns=[...])` projeksiyonu
C++ katmanında aşağı iter (ölçüldü: 1 kolon 0.43 GB / 5 kolon 1.15 GB — lineer). Market
3 kolon grubuna bölünür, gruplar BigQuery'de `row_id` üzerinden birleştirilir.

Bu pozisyonel join varsayımı iki yerden doğrulanmıştır: `tests/test_ingestion.py`
(sentetik tek-batch dosyayla round-trip) ve BigQuery'de 221.7M satırda
`sample_id`/`seconds_before_predict` uyuşmazlığı = **0**.

---

## Metrik: cosine similarity

`cos(y, ŷ) = Σyŷ / (‖y‖·‖ŷ‖)` — **ölçek-değişmez ama kaydırma-değişmez değildir.**

- Tahminleri sabitle çarpmak skoru değiştirmez → magnitude kalibrasyonuna efor harcanmaz.
- Sabit bias eklemek skoru **bozar**. Ampirik kanıt: sabit tahmin eden `mean` modeli
  **−0.0036** cosine üretiyor.
- Ensemble ağırlıkları grid search gerektirmez: y'nin model tahminlerinin span'ine dik
  izdüşümü optimaldir, o da **OLS çözümüdür** (`tests/test_ensemble.py` 200 rastgele
  ağırlık vektörüne karşı doğrular).

---

## Validation: walk-forward + embargo

Random split **kullanılmaz** — ardışık sample'ların pencereleri kesişebilir.

```
Fold 1: train ay 0–34 │ embargo │ val 36–40
...
Fold 5: train ay 0–58 │ embargo │ val 60–64
HOLD-OUT (dokunulmaz): ay 65–70
```

`assert_fold_integrity()` her fold'da veri üzerinde yeniden doğrular.
`tests/test_train_integrity.py` bozuk kurulumlar (random split, embargo ihlali,
hold-out sızıntısı) enjekte edip korumanın bunları **yakaladığını** kanıtlar.

---

## Sonuçlar (ara — %25 örneklem, son 2 fold, 400 ağaç)

| Model | cosine (ort) | std | dir. acc |
|---|---:|---:|---:|
| **ensemble** | **+0.1455** | 0.0084 | — |
| lightgbm | +0.1440 | 0.0088 | 0.550 |
| xgboost | +0.1398 | 0.0053 | 0.549 |
| ridge | +0.1241 | 0.0100 | 0.543 |
| zero | 0.0000 | — | — |
| mean | −0.0036 | 0.0010 | 0.522 |

Tam veri + tüm fold sonuçları `make train` ile üretilir.

---

## Kurulum ve çalıştırma

```bash
pip install -r requirements-dev.txt
make check                 # lint + test (canlı BigQuery veya veri GEREKTİRMEZ)
```

`configs/config.yaml` içindeki `paths.data_root` verinin nereye yazılacağını belirler.
**Varsayılan `C:/mscapital_data` — bilerek OneDrive dışında**, çünkü ara veri ~20 GB.

```bash
make ingest      # feather → parquet → BigQuery → staging
make features    # BigQuery feature katmanı + lokale indirme
make train       # walk-forward + MLflow
make api         # FastAPI  :8000
make streamlit   # Dashboard :8501
```

Docker ile:

```bash
docker compose up -d      # api :8000, streamlit :8501, mlflow :5000
```

### Kimlik bilgileri

- **Kaggle**: `~/.kaggle/kaggle.json`
- **GCP**: `configs/config.yaml` → `credentials.gcp_service_account`.
  Anahtar dosyası repo dışında tutulmalıdır.

---

## Maliyet

BigQuery veriyi ~5.4× sıkıştırır (`market_g2`: 11.57 GiB logical → 2.12 GiB physical).
Dört dataset **physical storage billing**'e alınmış, `mscapital_raw` staging kurulduktan
sonra düşürülmüştür → toplam ~8 GiB physical, **10 GiB ücretsiz kotanın altında**.
Sorgu tarafı aylık 1 TiB ücretsiz kotanın ~%13'ü. Batch load job'lar ücretsizdir.

---

## Proje yapısı

```
src/
  config.py              tek merkezden yol ve sabit yönetimi
  data/                  ingestion (kolon grubu) · bq_loader · staging
  features/              market (159) · order (82) · transaction (53) · assemble
  evaluation/            metrics (cosine) · temporal_validation · backtesting
  models/                baseline · lightgbm · xgboost · ensemble · train (CLI)
  inference/             predictor — API bunu kullanır, eğitim kodunu değil
api/main.py              FastAPI: /health /model-info /predict /batch-predict /reload
streamlit_app/           6 sayfalık dashboard
sql/                     BigQuery staging DDL
tests/                   85 test
```
