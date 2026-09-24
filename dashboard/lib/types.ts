export type DashboardPayload = {
  champion: {
    version: string;
    validation_metric: number | null;
    trained_at: string | null;
    data_cutoff: string | null;
  } | null;
  driftSeries: {
    computed_at: string;
    accuracy_overall: number;
    accuracy_rolling_24h: number | null;
    drop_pct: number | null;
    triggered_retrain: boolean;
  }[];
  accuracyByStation: { station_id: string; accuracy: number; samples: number }[];
  grid3d: {
    stations: string[];
    batches: string[];
    values: [number, number, number][]; // [stationIndex, batchIndex, accuracy]
  };
  leaderboard: {
    rank: number;
    accuracy: number;
    coverage: number;
    total: number;
  } | null;
  status: {
    collector: string | null;
    infer: string | null;
    drift: string | null;
    train: string | null;
  };
  totals: {
    evaluations: number;
    predictions: number;
    models: number;
  };
};
