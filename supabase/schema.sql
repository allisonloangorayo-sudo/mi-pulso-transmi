-- Pulso TransMi — esquema de la memoria operacional del equipo.
-- Ejecutar completo en Supabase > SQL Editor antes de correr src/ingest.py.
--
-- Diseño: las columnas cuyo nombre exacto no está confirmado por la API
-- (atributos de estación, campos de contexto) se guardan como jsonb en vez
-- de inventar nombres de columna, para no romper si la API cambia su forma.

create table if not exists stations (
    station_id text primary key,
    attributes jsonb not null default '{}'::jsonb,
    inserted_at timestamptz not null default now()
);

create table if not exists observations (
    station_id text not null references stations(station_id),
    observed_at timestamptz not null,
    demand double precision not null,
    inserted_at timestamptz not null default now(),
    primary key (station_id, observed_at)
);

create index if not exists idx_observations_observed_at on observations (observed_at);

create table if not exists context (
    observed_at timestamptz primary key,
    payload jsonb not null default '{}'::jsonb,
    inserted_at timestamptz not null default now()
);

-- Cursor persistido del collector incremental. Una fila por stream
-- ('observations' | 'context'). Solo avanza después de confirmar el upsert.
create table if not exists collector_cursor (
    stream text primary key,
    last_cursor text,
    last_observed_at timestamptz,
    updated_at timestamptz not null default now()
);

-- Bitácora de cada ejecución del collector, incluso sin novedades.
create table if not exists ingestion_runs (
    id bigint generated always as identity primary key,
    stream text not null,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    rows_fetched integer not null default 0,
    cursor_before text,
    cursor_after text,
    status text not null check (status in ('ok', 'no_new_data', 'error')),
    error_message text
);

-- Versiones de modelo: candidato, champion vigente, o histórico.
create table if not exists model_versions (
    version text primary key,
    trained_at timestamptz not null default now(),
    data_cutoff timestamptz not null,
    commit_sha text,
    features jsonb,
    validation_metric double precision,
    artifact_path text not null,
    status text not null default 'candidate'
        check (status in ('candidate', 'champion', 'historical'))
);

-- Predicciones emitidas. cycle_id queda null en pruebas locales fuera de
-- ventana competitiva; el batch real llenará station_id + target_at exactos.
create table if not exists predictions (
    id bigint generated always as identity primary key,
    cycle_id text,
    station_id text not null references stations(station_id),
    target_at timestamptz not null,
    model_version text not null references model_versions(version),
    value double precision not null,
    submission_id text,
    idempotency_key text,
    submitted_at timestamptz not null default now(),
    unique (cycle_id, station_id, target_at, model_version)
);

-- Evaluaciones: predicción vs. realidad, una vez se conoce el dato real.
--
-- `source` importa para el drift: las evaluaciones 'simulado' replican
-- historia que el modelo ya conoce y dan una lectura optimista; solo las
-- 'real' reflejan el desempeño en la competencia. Mezclarlas escondía la
-- degradación real.
create table if not exists evaluations (
    id bigint generated always as identity primary key,
    prediction_id bigint references predictions(id),
    station_id text not null references stations(station_id),
    target_at timestamptz not null,
    model_version text not null,
    predicted double precision not null,
    real double precision not null,
    abs_error double precision generated always as (abs(predicted - real)) stored,
    source text not null default 'simulado' check (source in ('real', 'simulado')),
    cycle_id text,
    evaluated_at timestamptz not null default now(),
    unique (station_id, target_at, model_version, source)
);

create index if not exists idx_evaluations_source on evaluations (source, evaluated_at desc);

-- Estado genérico del pipeline (ej. el "reloj" simulado que usa src/infer.py
-- mientras la API no libera ciclos reales). Una fila por clave.
create table if not exists pipeline_state (
    key text primary key,
    value jsonb,
    updated_at timestamptz not null default now()
);

-- Bitácora de cada ejecución de src/infer.py: qué hizo, con qué modelo y
-- cómo terminó. `ingestion_runs` cubre al collector; esta cubre la
-- inferencia, que antes no dejaba rastro salvo en los logs de Actions.
create table if not exists inference_runs (
    id bigint generated always as identity primary key,
    run_at timestamptz not null default now(),
    github_run_id text,                    -- identificador de la ejecución en Actions
    mode text not null                     -- qué camino tomó la corrida
        check (mode in ('real', 'simulado', 'sin_ciclo')),
    cycle_id text,
    data_cutoff timestamptz,               -- hasta dónde vio datos el predictor
    model_version text,
    predictions_count integer not null default 0,
    submission_id text,                    -- recibo de la API, si hubo envío
    status text not null check (status in ('ok', 'skipped', 'error')),
    error_message text,
    duration_ms integer
);

create index if not exists idx_inference_runs_run_at on inference_runs (run_at desc);

-- Resultados de cada corrida de detección de drift (accuracy acumulada vs.
-- rolling 24h y si esa corrida disparó un reentrenamiento automático).
create table if not exists drift_metrics (
    id bigint generated always as identity primary key,
    computed_at timestamptz not null default now(),
    sample_size integer not null,
    accuracy_overall double precision,
    accuracy_rolling_24h double precision,
    drop_pct double precision,
    triggered_retrain boolean not null default false
);
