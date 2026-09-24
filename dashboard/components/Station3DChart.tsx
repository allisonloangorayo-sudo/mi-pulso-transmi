"use client";

import ReactECharts from "echarts-for-react";
import * as echarts from "echarts";
import "echarts-gl";
import { palette } from "@/lib/palette";
import type { DashboardPayload } from "@/lib/types";

export function Station3DChart({ grid3d }: { grid3d: DashboardPayload["grid3d"] }) {
  const { stations, batches, values } = grid3d;

  const batchLabels = batches.map((b) =>
    new Date(b).toLocaleString("es-CO", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })
  );

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      formatter: (params: any) => {
        const [sIdx, bIdx, acc] = params.value;
        return `${stations[sIdx]}<br/>${batchLabels[bIdx]}<br/><b>${acc.toFixed(1)}%</b> accuracy`;
      },
    },
    visualMap: {
      show: true,
      dimension: 2,
      min: 0,
      max: 100,
      calculable: true,
      inRange: { color: palette.sequentialBlue as unknown as string[] },
      textStyle: { color: "var(--text-secondary)" },
      right: 8,
      top: "middle",
    },
    xAxis3D: {
      type: "category",
      data: stations,
      name: "Estación",
      axisLabel: { color: "var(--text-secondary)", fontSize: 10 },
      nameTextStyle: { color: "var(--text-secondary)" },
    },
    yAxis3D: {
      type: "category",
      data: batchLabels,
      name: "Ciclo",
      axisLabel: { color: "var(--text-secondary)", fontSize: 9 },
      nameTextStyle: { color: "var(--text-secondary)" },
    },
    zAxis3D: {
      type: "value",
      name: "Accuracy %",
      min: 0,
      max: 100,
      nameTextStyle: { color: "var(--text-secondary)" },
      axisLabel: { color: "var(--text-secondary)" },
    },
    grid3D: {
      boxWidth: 120,
      boxDepth: 90,
      viewControl: { autoRotate: true, autoRotateSpeed: 6, distance: 220 },
      light: {
        main: { intensity: 1.3, shadow: true },
        ambient: { intensity: 0.4 },
      },
    },
    series: [
      {
        type: "bar3D",
        data: values,
        shading: "lambert",
        bevelSize: 0.15,
        itemStyle: { opacity: 0.95 },
      },
    ],
  };

  if (values.length === 0) {
    return (
      <div className="flex h-96 items-center justify-center text-sm" style={{ color: "var(--text-secondary)" }}>
        Todavía no hay suficientes ciclos evaluados para el mapa 3D.
      </div>
    );
  }

  return (
    <ReactECharts
      echarts={echarts}
      option={option}
      style={{ height: 420, width: "100%" }}
      notMerge
      lazyUpdate
    />
  );
}
