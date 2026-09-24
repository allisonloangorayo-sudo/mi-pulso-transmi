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
create table if not exists evaluations (
    id bigint generated always as identity primary key,
    prediction_id bigint references predictions(id),
    station_id text not null references stations(station_id),
    target_at timestamptz not null,
    model_version text not null,
    predicted double precision not null,
    real double precision not null,
    abs_error double precision generated always as (abs(predicted - real)) stored,
    evaluated_at timestamptz not null default now()
);
