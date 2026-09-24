import { NextResponse } from "next/server";
import { getSupabaseServer } from "@/lib/supabase-server";
import type { DashboardPayload } from "@/lib/types";

export const dynamic = "force-dynamic";

type EvaluationRow = {
  station_id: string;
  predicted: number;
  real: number;
  abs_error: number | null;
  evaluated_at: string;
};

async function fetchLeaderboard(): Promise<DashboardPayload["leaderboard"]> {
  const apiUrl = process.env.PULSO_API_URL || "https://pulso-transmi.72-60-245-2.sslip.io";
  const apiKey = process.env.PULSO_API_KEY;
  if (!apiKey) return null;
  try {
    const res = await fetch(`${apiUrl}/v1/leaderboard`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      cache: "no-store",
    });
    if (!res.ok) return null;
    const data = await res.json();
    const rows: any[] = data.data ?? [];
    const participantName = process.env.PULSO_PARTICIPANT_NAME || "";
    const me = rows.find((row) => participantName && row.display_name?.includes(participantName));
    if (!me) return null;
    return { rank: me.rank, accuracy: me.accuracy, coverage: me.coverage, total: data.count ?? rows.length };
  } catch {
    return null;
  }
}

export async function GET() {
  try {
    const supabase = getSupabaseServer();

    const [driftRes, evalRes, modelsRes, runsRes, predRes] = await Promise.all([
      supabase.from("drift_metrics").select("*").order("computed_at", { ascending: true }).limit(500),
      supabase
        .from("evaluations")
        .select("station_id, predicted, real, abs_error, evaluated_at")
        .order("evaluated_at", { ascending: true })
        .limit(5000),
      supabase.from("model_versions").select("*").order("trained_at", { ascending: false }).limit(50),
      supabase.from("ingestion_runs").select("*").order("started_at", { ascending: false }).limit(30),
      supabase.from("predictions").select("submitted_at").order("submitted_at", { ascending: false }).limit(1),
    ]);

    const drift = driftRes.data ?? [];
    const evaluations = (evalRes.data ?? []) as EvaluationRow[];
    const models = modelsRes.data ?? [];
    const runs = runsRes.data ?? [];

    const champion = models.find((m: any) => m.status === "champion") ?? null;

    // Accuracy agregado por estación
    const byStation = new Map<string, { real: number; abs: number; samples: number }>();
    for (const row of evaluations) {
      const agg = byStation.get(row.station_id) ?? { real: 0, abs: 0, samples: 0 };
      agg.real += row.real;
      agg.abs += row.abs_error ?? Math.abs(row.predicted - row.real);
      agg.samples += 1;
      byStation.set(row.station_id, agg);
    }
    const accuracyByStation = Array.from(byStation.entries())
      .map(([station_id, agg]) => ({
        station_id,
        accuracy: agg.real > 0 ? Math.max(0, 100 * (1 - agg.abs / agg.real)) : 0,
        samples: agg.samples,
      }))
      .sort((a, b) => a.station_id.localeCompare(b.station_id));

    // Grid 3D: estación x lote (evaluated_at compartido por las 48 filas de
    // un mismo ciclo) -> accuracy. Se limita a los últimos 12 lotes.
    const batchSet = Array.from(new Set(evaluations.map((e) => e.evaluated_at))).sort();
    const recentBatches = batchSet.slice(-12);
    const stations = accuracyByStation.map((s) => s.station_id);

    const values: [number, number, number][] = [];
    recentBatches.forEach((batch, batchIdx) => {
      stations.forEach((stationId, stationIdx) => {
        const rows = evaluations.filter((e) => e.evaluated_at === batch && e.station_id === stationId);
        if (rows.length === 0) return;
        const real = rows.reduce((s, r) => s + r.real, 0);
        const abs = rows.reduce((s, r) => s + (r.abs_error ?? Math.abs(r.predicted - r.real)), 0);
        const accuracy = real > 0 ? Math.max(0, 100 * (1 - abs / real)) : 0;
        values.push([stationIdx, batchIdx, Number(accuracy.toFixed(1))]);
      });
    });

    const lastObservationRun = runs.find((r: any) => r.stream === "observations");
    const lastDrift = drift.length ? drift[drift.length - 1] : null;
    const lastEval = evaluations.length ? evaluations[evaluations.length - 1] : null;

    const payload: DashboardPayload = {
      champion: champion
        ? {
            version: champion.version,
            validation_metric: champion.validation_metric,
            trained_at: champion.trained_at,
            data_cutoff: champion.data_cutoff,
          }
        : null,
      driftSeries: drift.map((d: any) => ({
        computed_at: d.computed_at,
        accuracy_overall: d.accuracy_overall,
        accuracy_rolling_24h: d.accuracy_rolling_24h,
        drop_pct: d.drop_pct,
        triggered_retrain: d.triggered_retrain,
      })),
      accuracyByStation,
      grid3d: { stations, batches: recentBatches, values },
      leaderboard: await fetchLeaderboard(),
      status: {
        collector: lastObservationRun?.finished_at ?? lastObservationRun?.started_at ?? null,
        infer: lastEval?.evaluated_at ?? predRes.data?.[0]?.submitted_at ?? null,
        drift: lastDrift?.computed_at ?? null,
        train: champion?.trained_at ?? null,
      },
      totals: {
        evaluations: evaluations.length,
        predictions: predRes.data ? predRes.count ?? 0 : 0,
        models: models.length,
      },
    };

    return NextResponse.json(payload);
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
}
