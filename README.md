# Mi Pulso TransMi

Proyecto de equipo — reto MLOps Pulso TransMi (Universidad Externado de Colombia).
Construido sobre el [SDK oficial del profesor](https://github.com/uexternadojz/pulso-transmi-sdk).

## Estado actual (2026-09-24)

La API pública sigue en modo **solo lectura** (`0.2.0`): expone histórico
estático (45 días, 12 estaciones, 51.840 observaciones), pero **todavía no
publica** `/v1/forecast-cycles/current` ni el endpoint de submissions. Estamos
en la **Fase 1 — datos estáticos** del taller, no en la ventana competitiva.

Por eso `src/predict.py` (contrato oficial) queda como scaffold con las
partes bloqueadas por la API marcadas `TODO`, mientras que `src/infer.py`
(el que corre en producción, cada hora) opera en **modo simulado**: avanza un
reloj propio sobre el histórico ya conocido, genera las 48 predicciones y las
evalúa de inmediato contra el dato real (que ya tenemos, solo que aún no "ha
pasado" para el reloj simulado). Esto da accuracy y drift medibles *hoy*, sin
esperar a que el profesor active la ventana competitiva. En cuanto
`/v1/forecast-cycles/current` responda con un ciclo real, `infer.py` lo
detecta solo y cambia automáticamente al flujo oficial de `predict.py`.

Techo de accuracy observado: el candidato satura alrededor de **86-89** de
accuracy (WAPE invertido). Es esperable — la demanda es sintética con ruido
y ese es probablemente el piso irreducible del generador, no un problema del
pipeline. `drift.py` vigila caídas *relativas* a ese nivel, no una meta fija.

## Arquitectura

```text
src/
├── ingest.py    # collector idempotente: descarga y upsert a Supabase
├── features.py  # calendario + rezagos (lags)
├── train.py     # baselines + candidato + promoción automática a champion
├── infer.py     # inferencia horaria: ciclo real si existe, si no simulación
├── predict.py   # contrato oficial de submissions (scaffold, pendiente de API)
├── drift.py     # accuracy acumulada vs. rolling 24h + dispara reentrenamiento
├── monitor.py   # reporte de accuracy legible (offline + Supabase)
└── db.py        # cliente Supabase: upserts, cursor, bitácora, Storage del modelo
supabase/schema.sql   # DDL de la memoria operacional (9 tablas)
.github/workflows/
├── collector.yml  # cada hora: sincroniza datos a Supabase
├── infer.yml      # cada hora: 48 predicciones (12 estaciones x 4 horizontes)
├── drift.yml      # cada 2 horas: accuracy/drift; dispara train.yml si cae ≥5 pts
└── train.yml      # manual + disparado automáticamente por drift.yml
```

### Las 4 GitHub Actions

| Acción | Cadencia | Qué hace |
|---|---|---|
| `collector.yml` | cada hora | Descarga y sincroniza a Supabase (idempotente). Hoy el histórico es estático, así que cada corrida re-sincroniza lo mismo — inofensivo por el upsert, y queda listo para cuando la API empiece a liberar datos nuevos de verdad. |
| `infer.yml` | cada hora | Genera 48 predicciones (12 estaciones × 4 horizontes: +15/+30/+45/+60 min) con el champion vigente y las evalúa. |
| `drift.yml` | cada 2 horas | Calcula accuracy acumulada vs. rolling 24h y la guarda en `drift_metrics`. Si la caída llega a **5 puntos**, dispara `train.yml` automáticamente (`gh workflow run`). |
| `train.yml` | manual + automático (por drift) | Entrena un candidato, lo compara contra el champion vigente y **lo promueve automáticamente si lo supera** (nunca si no). |

Con el histórico estático de hoy, la caída de accuracy debería mantenerse
cerca de 0 casi siempre — `train.yml` no debería dispararse solo todavía.
Eso es lo esperado: el mecanismo ya queda automático y listo, pero solo
actúa cuando el drift es real (cuando cambien las tendencias, como pediste).

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

Automática desde `train.py` (vía `db.promote_if_better`): un candidato
reemplaza al champion **únicamente si lo supera** en la métrica de
validación temporal — "la novedad por sí sola no es una mejora". Si nunca ha
habido champion, el primer candidato válido lo es. El artefacto se sube a
Supabase Storage (bucket `models`) para que cualquier runner de Actions
pueda descargarlo, sin importar cuál lo entrenó.

Para forzar manualmente una versión distinta (poco común, solo si necesitas
un rollback):

```sql
update model_versions set status = 'historical' where status = 'champion';
update model_versions set status = 'champion' where version = '<version_elegida>';
```

## GitHub Actions

Los 4 workflows están activos con `schedule` real (`collector.yml`,
`infer.yml`, `drift.yml`) o disparo automático (`train.yml`). Antes de que
corran de verdad necesitan los secrets del repositorio:

```bash
gh secret set SUPABASE_URL --body "https://xxxx.supabase.co"
gh secret set SUPABASE_KEY --body "eyJ..."
# cuando se habilite la competencia:
gh secret set PULSO_API_KEY --body "..."
```

`GITHUB_TOKEN` (para que `drift.yml` dispare `train.yml`) ya lo provee
GitHub automáticamente — no hay que crearlo. Nunca subas las claves de
arriba al código ni las imprimas en logs.

## Próximos pasos (cuando se habilite la ventana competitiva)

- [ ] Reemplazar el `TODO` de `submit_predictions()` en `src/predict.py` con
      la ruta y payload reales del contrato técnico publicado.
- [ ] Cuando `/v1/forecast-cycles/current` empiece a responder, `infer.py` va
      a delegar automáticamente en `src.predict` — verificar el primer ciclo
      real con cuidado (revisar logs de `infer.yml`).
- [ ] Ajustar la cadencia de `collector.yml`/`infer.yml` a la que anuncie el
      profesor (la guía operativa sugiere cada 10 min una vez activa la
      competencia; hoy corren cada hora porque no hay datos nuevos que
      justifiquen más frecuencia).
- [ ] Conectar el bono de dashboard (Vercel) a `drift_metrics` y
      `evaluations` para visualizar la serie de accuracy en tiempo real.
