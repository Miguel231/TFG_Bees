# Swarm Risk API

API local (FastAPI) que envuelve el mejor modelo del TFG — LSTM unidireccional de
`03_swarm_night_enhanced.ipynb` (AUC=0.887, 13/14 eventos de test detectados, 218 falsas alarmas).

## 1. Generar los artefactos del modelo

Abre `notebooks/models/03_swarm_night_enhanced.ipynb` y ejecuta hasta el final de la
**Sección 7.3 "Exportar el modelo para inferencia"**. Esto crea una carpeta `api_export/`
en la raíz del repo con 5 archivos:

```
api_export/
├── lstm_uni_03.pt        # pesos del LSTM
├── scaler_03.pkl         # StandardScaler ajustado en train
├── feat_a_03.json        # lista de features y su orden exacto
├── median_fill_03.json   # medianas de imputación (train)
└── model_meta_03.json    # hiperparámetros de arquitectura
```

Copia esos 5 archivos a `api/models/` (la API los busca ahí por defecto):

```bash
mkdir api/models
cp api_export/* api/models/
```

## 2. Instalar y arrancar

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Si todo está bien, en `http://localhost:8000/health` debería responder `{"status": "ok"}`.
Si falta algún artefacto, lo dirá explícitamente con la lista de archivos que faltan.

## 3. Probar una predicción

El endpoint espera un CSV con las columnas del dataset unificado (`Hive name`, `Time`,
`Weight`, `Frequency`, `Volume`, `Temperature heart`, `Humidity heart`, `Temperature scale`,
`Humidity scale`) con **al menos ~35 días** de histórico hasta la fecha más reciente —
cuantos más días mejor, ya que varias features son rolling de 14 días.

```bash
curl -X POST "http://localhost:8000/predict?box_id=3" \
     -F "file=@ultimos_45_dias_box3.csv"
```

Respuesta de ejemplo:

```json
{
  "box_id": 3,
  "date": "2026-06-10",
  "swarm_risk_probability": 0.34,
  "risk_level": "MEDIO",
  "horizon_days": 3,
  "message": "Riesgo de enjambrazon en los proximos 3 dias: MEDIO (34.0%)"
}
```

`risk_level`: `BAJO` (<20%), `MEDIO` (20–50%), `ALTO` (≥50%). Estos umbrales son provisionales —
ajústalos cuando tengas más eventos reales con los que validar el comportamiento del sistema en
producción.

## Limitaciones conocidas (para la memoria / defensa)

- **`days_since_swarm` / `has_prior_swarm`** se fijan a un valor centinela (999 / 0) en
  producción porque la API no mantiene un histórico de enjambrazones confirmados — en el
  notebook de entrenamiento sí se conocía esa información. Si se integra una base de datos
  de eventos confirmados, esta función debería actualizarse para usarla.
- El pipeline de features se ha **reescrito a mano** en `inference.py` replicando el notebook
  línea a línea — si se modifica la ingeniería de features en el notebook, hay que actualizar
  `inference.py` igual, no se reutiliza código automáticamente entre ambos.
- No hay capa de base de datos: cada llamada a `/predict` recibe el CSV completo y recalcula
  todo desde cero. Para producción real con muchas colmenas, conviene cachear el `feat_day`
  ya calculado y solo añadir el día nuevo en cada actualización de n8n.
