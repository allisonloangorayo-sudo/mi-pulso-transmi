# Mi Pulso TransMi

Proyecto de equipo — reto MLOps Pulso TransMi (Universidad Externado de Colombia).
Construido sobre el [SDK oficial del profesor](https://github.com/uexternadojz/pulso-transmi-sdk).

## Estado actual (2026-09-24, actualizado 03:55 UTC)

**La ventana competitiva está activa.** La API subió de `0.2.0` a `0.7.1`,
`/v1/clock` reporta `state: running`, y `/v1/forecast-cycles/current` +
`/v1/submissions` + `/v1/leaderboard` ya funcionan de verdad. Confirmado en
producción: nuestra primera submission real fue aceptada (`201`, 48/48
predicciones, `is_official: true`).

Detalle importante que descubrimos en vivo: `/v1/observations` solo sirve el
histórico estático original (se congela en `history_end`); los datos nuevos
de la ventana competitiva **solo** llegan por `/v1/stream/observations`
(paginado con cursor). Tanto `src/ingest.py` como `src/predict.py` ya
combinan ambas fuentes.

`src/infer.py` es el punto de entrada real: intenta el ciclo oficial primero
(`GET /v1/forecast-cycles/current`) y delega en `src/predict.py` si hay uno
abierto; si no (por ejemplo, entre ciclos), cae a un modo simulado sobre el
histórico ya conocido para seguir generando accuracy/drift medible sin
depender de que haya un ciclo abierto en ese instante exacto.

Los ciclos reales duran solo **25 minutos** y no están alineados al reloj de
pared (el primero abrió a las `:37`, no a `:00`) — por eso `collector.yml` e
`infer.yml` corren cada **10 minutos**, como recomienda la guía operativa,
en vez de una vez por hora.

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
├── infer.py     # punto de entrada real: ciclo oficial si existe, si no simulación
├── predict.py   # contrato oficial de submissions (ya en producción)
├── drift.py     # accuracy acumulada vs. rolling 24h + dispara reentrenamiento
├── monitor.py   # reporte de accuracy legible (offline + Supabase)
└── db.py        # cliente Supabase: upserts, cursor, bitácora, Storage del modelo
supabase/schema.sql   # DDL de la memoria operacional (9 tablas)
.github/workflows/
├── collector.yml  # cada 10 min: sincroniza histórico + stream a Supabase
├── infer.yml      # cada 10 min: 48 predicciones (12 estaciones x 4 horizontes) + submission real
├── drift.yml      # cada hora: accuracy/drift; dispara train.yml si cae ≥5 pts
└── train.yml      # manual + disparado automáticamente por drift.yml
```

### Las 4 GitHub Actions

| Acción | Cadencia | Qué hace |
|---|---|---|
| `collector.yml` | cada 10 min | Sincroniza histórico + stream incremental a Supabase (idempotente). |
| `infer.yml` | cada 10 min | Si hay ciclo real abierto: genera 48 predicciones y las envía a `/v1/submissions`. Si no: simula sobre el histórico para seguir midiendo accuracy/drift. |
| `drift.yml` | cada hora | Calcula accuracy acumulada vs. rolling 24h y la guarda en `drift_metrics`. Si la caída llega a **5 puntos**, dispara `train.yml` automáticamente (`gh workflow run`). |
| `train.yml` | manual + automático (por drift) | Entrena un candidato, lo compara contra el champion vigente y **lo promueve automáticamente si lo supera** (nunca si no). |

`collector.yml` e `infer.yml` corren cada 10 minutos porque los ciclos reales
duran solo 25 minutos y no están alineados al reloj — con menos frecuencia
se corre el riesgo real de perder la ventana de un ciclo.

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

### El `schedule:` nativo no funcionó — disparo externo

El `schedule:` de GitHub Actions **nunca se disparó** en este repositorio
(verificado con `GET /actions/runs?event=schedule` → `total_count: 0` durante
más de 2 horas, con los workflows en estado `active`, Actions habilitado,
cron válido en la rama por defecto y tras forzar un `disable`/`enable`).
Es una falla conocida del scheduler de GitHub con workflows recién creados
cuyo cron se edita varias veces seguidas.

Solución en producción: **cron-job.org** llama al endpoint
`POST /repos/{owner}/{repo}/actions/workflows/{id}/dispatches` de la API de
GitHub con un token fine-grained (permiso *Actions: read and write*, alcance
limitado a este repo):

| Job externo | Cadencia | Workflow disparado |
|---|---|---|
| `pulso-transmi-collector` | minutos `:00 :10 :20 :30 :40 :50` | `collector.yml` |
| `pulso-transmi-infer` | minutos `:05 :15 :25 :35 :45 :55` | `infer.yml` |
| `pulso-transmi-drift` | minuto `:50` de cada hora | `drift.yml` |

Los `schedule:` nativos se dejaron declarados en los YAML por si GitHub los
reactiva; un disparo doble es inofensivo porque los workflows son
idempotentes (`receipt_exists`, upsert por llave primaria).

## Dashboard (bono)

`dashboard/` es una app Next.js desplegada en Vercel:
**https://mi-pulso-transmi-dashboard.vercel.app**

Lee Supabase y el leaderboard oficial **solo desde el servidor** (la ruta
`app/api/dashboard/route.ts`), así la `service_role` key y la `PULSO_API_KEY`
nunca llegan al navegador — requisito explícito de la guía. Paneles:
accuracy acumulada vs. rolling 24h, mapa 3D de accuracy por estación×ciclo,
accuracy por estación, y el estado de la última corrida de cada workflow.

```bash
cd dashboard
npm install
cp .env.local.example .env.local   # completar con las claves
npm run dev
```

## Estado de la ventana competitiva

- [x] `submit_predictions()` implementado contra el endpoint real (`POST
      /v1/submissions`, esquema confirmado: `schema_version`, `cycle_id`,
      `client_run_id`, `data_cutoff`, `model`, `predictions[]`).
- [x] Primera submission oficial aceptada (`201`, 48/48, `is_official: true`).
- [x] `collector.yml`/`infer.yml` corren cada 10 minutos.
- [ ] Ver la posición en `/v1/leaderboard` mejora a medida que la API resuelve
      los targets (los horizontes se evalúan progresivamente, ver guía
      operativa p.9) y a medida que se acumulan más ciclos entregados.
- [ ] Conectar el bono de dashboard (Vercel) a `drift_metrics`, `evaluations`
      y al leaderboard oficial para visualizar todo en tiempo real.
- [ ] Revisar si conviene ampliar `build_features_as_of` para no repetir el
      mismo bloque de lags en los 4 horizontes (mejora de precisión, no de
      correctitud — el envío ya es válido tal como está).
