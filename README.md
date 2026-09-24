# Mi Pulso TransMi

Proyecto de equipo — reto MLOps Pulso TransMi (Universidad Externado de Colombia).
Construido sobre el [SDK oficial del profesor](https://github.com/uexternadojz/pulso-transmi-sdk).

## Estado actual (2026-09-23)

La API pública sigue en modo **solo lectura** (`0.2.0`): expone histórico
estático (45 días, 12 estaciones, 51.840 observaciones), pero **todavía no
publica** `/v1/forecast-cycles/current` ni el endpoint de submissions. Estamos
en la **Fase 1 — datos estáticos** del taller, no en la ventana competitiva.

Por eso `src/predict.py` es un scaffold fiel al contrato documentado (guía
operativa v2.0), con las partes bloqueadas por la API marcadas `TODO`. Todo lo
demás (`ingest`, `features`, `train`, `monitor`) ya funciona con datos reales
hoy.

## Arquitectura

```text
src/
├── ingest.py    # collector idempotente: descarga y upsert a Supabase
├── features.py  # calendario + rezagos (lags)
├── train.py     # baselines + candidato con validación temporal
├── predict.py   # inferencia por ciclo (scaffold, pendiente de API)
├── monitor.py   # accuracy, rolling 24h, señal de drift
└── db.py        # cliente Supabase + upserts + cursor + bitácora
supabase/schema.sql   # DDL de la memoria operacional
.github/workflows/
├── predict.yml  # inferencia y submission (cron pendiente de activar)
└── train.yml    # entrenamiento/promoción, workflow separado
```

## Puesta en marcha

### 1. Entorno local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Crear el proyecto de Supabase

1. Crea una cuenta en [supabase.com](https://supabase.com) y un proyecto nuevo.
2. En **Project Settings → API**, copia `Project URL` y la clave `service_role`
   (no la `anon`: el collector necesita permisos de escritura desde un entorno
   de servidor/CI, nunca desde el navegador).
3. Pega esos valores en tu `.env` local:
   ```
   SUPABASE_URL=https://xxxx.supabase.co
   SUPABASE_KEY=eyJ...   # service_role
   ```
4. En **SQL Editor**, pega y ejecuta todo el contenido de
   [`supabase/schema.sql`](supabase/schema.sql). Esto crea las tablas de
   estaciones, observaciones, contexto, cursor del collector, bitácora de
   ingestas, versiones de modelo, predicciones y evaluaciones.

### 3. Cargar el histórico a Supabase

```bash
python -m src.ingest
```

Es **idempotente**: correrlo varias veces no duplica filas (upsert por
`station_id + observed_at`). Imprime un reporte de calidad (duplicados,
nulos, negativos, cobertura por estación) antes de escribir.

### 4. Entrenar baselines + candidato

```bash
python -m src.train
```

Usa partición **temporal** (últimos 7 días como validación, nunca aleatoria).
Compara el candidato (RandomForest) contra dos baselines (naive 24h y
estacional 7 días) y guarda el artefacto en `artifacts/` + metadata JSON.
Si hay credenciales de Supabase, registra la versión en `model_versions`
como `candidate`.

### 5. Pruebas

```bash
pytest -q
```

## Promoción del champion

`train.py` nunca promueve automáticamente. Para convertir un candidato en
champion (regla de la guía: "la novedad por sí sola no es una mejora"):

```sql
update model_versions set status = 'historical' where status = 'champion';
update model_versions set status = 'champion' where version = '<version_elegida>';
```

## GitHub Actions

Los dos workflows (`predict.yml`, `train.yml`) están en `workflow_dispatch`
manual con el `schedule` comentado a propósito — actívalo solo cuando el
profesor confirme la frecuencia oficial (la guía recomienda cada 10 min para
inferencia). Antes de eso, configura los secrets del repositorio:

```bash
gh secret set SUPABASE_URL --body "https://xxxx.supabase.co"
gh secret set SUPABASE_KEY --body "eyJ..."
# cuando se habilite la competencia:
gh secret set PULSO_API_KEY --body "..."
```

Nunca subas estas claves al código ni las imprimas en logs.

## Próximos pasos (cuando se habilite la ventana competitiva)

- [ ] Reemplazar el `TODO` de `submit_predictions()` en `src/predict.py` con
      la ruta y payload reales del contrato técnico publicado.
- [ ] Guardar el artefacto del champion en Supabase Storage o GitHub Release
      (hoy `predict.py` asume que el runner tiene el archivo local, lo cual
      no es cierto en un runner limpio de Actions).
- [ ] Activar el `schedule` de `predict.yml` con la frecuencia anunciada.
- [ ] Conectar `monitor.py` a las evaluaciones reales una vez existan
      submissions evaluadas.
