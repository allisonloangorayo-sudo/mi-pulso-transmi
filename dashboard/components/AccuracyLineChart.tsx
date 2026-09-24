"use client";

import ReactECharts from "echarts-for-react";
import { palette } from "@/lib/palette";
import type { DashboardPayload } from "@/lib/types";

export function AccuracyLineChart({ series }: { series: DashboardPayload["driftSeries"] }) {
  const dates = series.map((d) => new Date(d.computed_at));
  const overall = series.map((d) => d.accuracy_overall);
  const rolling = series.map((d) => d.accuracy_rolling_24h ?? null);

  const option = {
    backgroundColor: "transparent",
    grid: { left: 48, right: 24, top: 40, bottom: 40 },
    legend: {
      top: 0,
      left: 0,
      textStyle: { color: "var(--text-secondary)" },
      data: ["Accuracy acumulada", "Rolling 24h"],
    },
    tooltip: {
      trigger: "axis",
      valueFormatter: (v: number) => (v == null ? "—" : `${v.toFixed(1)}%`),
    },
    xAxis: {
      type: "category",
      data: dates.map((d) => d.toLocaleString("es-CO", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })),
      axisLine: { lineStyle: { color: palette.light.baseline } },
      axisLabel: { color: "var(--text-secondary)", fontSize: 10 },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 100,
      splitLine: { lineStyle: { color: "var(--gridline)" } },
      axisLabel: { color: "var(--text-secondary)", formatter: "{value}%" },
    },
    series: [
      {
        name: "Accuracy acumulada",
        type: "line",
        data: overall,
        showSymbol: false,
        lineStyle: { width: 2, color: palette.categorical.blue.light },
        itemStyle: { color: palette.categorical.blue.light },
      },
      {
        name: "Rolling 24h",
        type: "line",
        data: rolling,
        showSymbol: false,
        lineStyle: { width: 2, color: palette.categorical.orange.light },
        itemStyle: { color: palette.categorical.orange.light },
      },
    ],
  };

  if (series.length === 0) {
    return <EmptyState message="Todavía no hay corridas de drift.py registradas." />;
  }

  return <ReactECharts option={option} style={{ height: 320, width: "100%" }} notMerge lazyUpdate />;
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex h-80 items-center justify-center text-sm" style={{ color: "var(--text-secondary)" }}>
      {message}
    </div>
  );
}
