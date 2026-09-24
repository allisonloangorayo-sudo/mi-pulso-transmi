"use client";

import ReactECharts from "echarts-for-react";
import { palette } from "@/lib/palette";
import type { DashboardPayload } from "@/lib/types";

export function StationBarChart({ data }: { data: DashboardPayload["accuracyByStation"] }) {
  const sorted = [...data].sort((a, b) => b.accuracy - a.accuracy);

  const option = {
    backgroundColor: "transparent",
    grid: { left: 70, right: 40, top: 16, bottom: 24 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      valueFormatter: (v: number) => `${v.toFixed(1)}%`,
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 100,
      splitLine: { lineStyle: { color: "var(--gridline)" } },
      axisLabel: { color: "var(--text-secondary)", formatter: "{value}%" },
    },
    yAxis: {
      type: "category",
      data: sorted.map((s) => s.station_id),
      axisLine: { lineStyle: { color: palette.light.baseline } },
      axisLabel: { color: "var(--text-secondary)" },
    },
    series: [
      {
        type: "bar",
        data: sorted.map((s) => s.accuracy),
        barWidth: "60%",
        itemStyle: { color: palette.categorical.blue.light, borderRadius: [0, 4, 4, 0] },
        label: {
          show: true,
          position: "right",
          formatter: (p: any) => `${p.value.toFixed(1)}%`,
          color: "var(--text-secondary)",
        },
      },
    ],
  };

  if (data.length === 0) {
    return (
      <div className="flex h-80 items-center justify-center text-sm" style={{ color: "var(--text-secondary)" }}>
        Sin evaluaciones todavía por estación.
      </div>
    );
  }

  return <ReactECharts option={option} style={{ height: Math.max(280, sorted.length * 26), width: "100%" }} notMerge lazyUpdate />;
}
