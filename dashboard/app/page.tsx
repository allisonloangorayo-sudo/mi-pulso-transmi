"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { StatTile, timeAgo } from "@/components/StatTile";
import type { DashboardPayload } from "@/lib/types";

// Los charts tocan window/canvas: se cargan solo en cliente.
const AccuracyLineChart = dynamic(
  () => import("@/components/AccuracyLineChart").then((m) => m.AccuracyLineChart),
  { ssr: false }
);
const StationBarChart = dynamic(
  () => import("@/components/StationBarChart").then((m) => m.StationBarChart),
  { ssr: false }
);
const Station3DChart = dynamic(
  () => import("@/components/Station3DChart").then((m) => m.Station3DChart),
  { ssr: false }
);

export default function Home() {
  const [data, setData] = useState<DashboardPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch("/api/dashboard", { cache: "no-store" });
        if (!res.ok) throw new Error((await res.json()).error ?? "Error desconocido");
        const json = await res.json();
        if (!cancelled) {
          setData(json);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    }
    load();
    const interval = setInterval(load, 60_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen" style={{ background: "var(--page)" }}>
      <header
        className="border-b px-6 py-5"
        style={{ background: "var(--brand-green)", borderColor: "var(--border)" }}
      >
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--brand-gold)" }}>
              Pulso TransMi · MLOps
            </p>
            <h1 className="text-2xl font-bold text-white">Panel operativo</h1>
          </div>
          {data?.leaderboard && (
            <div className="text-right text-white">
              <div className="text-3xl font-bold">#{data.leaderboard.rank}</div>
              <div className="text-xs opacity-80">de {data.leaderboard.total} participantes</div>
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {error && (
          <div
            className="mb-6 rounded-lg border p-4 text-sm"
            style={{ borderColor: "var(--status-critical)", color: "var(--status-critical)" }}
          >
            {error}
          </div>
        )}

        {!data && !error && <p style={{ color: "var(--text-secondary)" }}>Cargando datos de Supabase…</p>}

        {data && (
          <>
            <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <StatTile
                label="Champion"
                value={data.champion?.version.replace("v", "") ?? "—"}
                sub={data.champion ? `accuracy val. ${data.champion.validation_metric?.toFixed(1)}%` : "sin champion"}
              />
              <StatTile
                label="Accuracy oficial"
                value={data.leaderboard ? `${data.leaderboard.accuracy.toFixed(1)}%` : "—"}
                sub={data.leaderboard ? `cobertura ${(data.leaderboard.coverage * 100).toFixed(0)}%` : "sin leaderboard"}
                tone={data.leaderboard && data.leaderboard.accuracy > 50 ? "good" : "warning"}
              />
              <StatTile label="Evaluaciones" value={String(data.totals.evaluations)} sub="filas en evaluations" />
              <StatTile
                label="Última inferencia"
                value={timeAgo(data.status.infer)}
                sub={data.status.infer ? new Date(data.status.infer).toLocaleTimeString("es-CO") : undefined}
              />
            </section>

            <section className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
              <StatTile label="Collector" value={timeAgo(data.status.collector)} />
              <StatTile label="Drift" value={timeAgo(data.status.drift)} />
              <StatTile label="Train" value={timeAgo(data.status.train)} />
              <StatTile
                label="Último drift"
                value={
                  data.driftSeries.length
                    ? `${data.driftSeries[data.driftSeries.length - 1].drop_pct?.toFixed(1)} pts`
                    : "—"
                }
                tone={
                  data.driftSeries.length && (data.driftSeries[data.driftSeries.length - 1].drop_pct ?? 0) >= 5
                    ? "critical"
                    : "good"
                }
              />
            </section>

            <ChartCard title="Accuracy acumulada vs. rolling 24h">
              <AccuracyLineChart series={data.driftSeries} />
            </ChartCard>

            <ChartCard title="Mapa 3D · accuracy por estación y ciclo">
              <Station3DChart grid3d={data.grid3d} />
            </ChartCard>

            <ChartCard title="Accuracy por estación (acumulado)">
              <StationBarChart data={data.accuracyByStation} />
            </ChartCard>
          </>
        )}
      </main>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section
      className="mt-6 rounded-xl border p-5"
      style={{ background: "var(--surface)", borderColor: "var(--border)" }}
    >
      <h2 className="mb-3 text-sm font-semibold" style={{ color: "var(--text-secondary)" }}>
        {title}
      </h2>
      {children}
    </section>
  );
}
