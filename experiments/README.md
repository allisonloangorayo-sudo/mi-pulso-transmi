# Experimentos

Evidencia de las decisiones de modelado (Fase 3 del taller). Todos evalúan
con la **métrica oficial** (WAPE → accuracy por estación, promedio simple),
sobre el **mismo conjunto de validación** y los **4 horizontes**, simulando
el servicio real: para un target `t` y horizonte `h`, solo se usan datos
`<= t-h`.

Los scripts importan rutas absolutas del entorno donde se corrieron; son un
registro de lo ejecutado, no parte del pipeline de producción.

## `01_skew_y_variantes.py`

Aísla el train/serve skew y prueba las variantes base.

| Variante | Accuracy | Por horizonte (+15/+30/+45/+60) |
|---|---|---|
| Producción inicial | 79.24% | 85.9 / 82.8 / 77.3 / 70.9 |
| + entrenar con el stream | 79.24% | idéntico* |
| + un modelo por horizonte | 84.60% | 85.9 / 85.1 / 84.2 / 83.1 |
| + estación como feature | 85.83% | 86.2 / 85.9 / 85.8 / 85.5 |
| + contexto (clima/eventos) | 85.27% | — |
| + boosting con pérdida MAE | 85.44% | — |

\* El diseño del holdout entrena siempre con datos anteriores a la ventana de
validación, así que los datos del stream nunca entran al entrenamiento en
esta comparación. Incluirlos sigue siendo correcto (es información más
reciente), pero su beneficio no es medible con este montaje.

**Conclusión**: el skew costaba ~5 puntos y se concentraba en los horizontes
largos. El contexto no ayuda porque solo cubre el periodo estático.

## `02_features_y_por_estacion.py`

| Variante | Accuracy |
|---|---|
| Base (por horizonte + estación) | 85.83% |
| + features ricas | 85.74% |
| + RF más profundo | 85.89% |
| **Modelos por estación** | **86.63%** |
| Ensamble RF+boosting | 86.18% |

**Conclusión**: separar por estación es lo que más aporta; las features
ricas y la profundidad extra están dentro del ruido.

## `03_reducir_modelos.py`

El problema práctico: 48 modelos (12 estaciones × 4 horizontes) pesan entre
282 MB (RF) y 1.8 GB (RF profundo), y `infer` descarga el artefacto cada 10
minutos. Se prueba apilar los 4 horizontes con `horizon` como feature para
quedarse con 12 modelos.

| Variante | Accuracy | Artefactos |
|---|---|---|
| 12 modelos, RF | 86.57% | 241.7 MB |
| **12 modelos, boosting MAE** | **86.69%** | **11.8 MB** |

**Conclusión (y configuración adoptada)**: empata con el mejor ensamble
(86.74%) pesando 99% menos. Comprimido en disco quedan 4.8 MB.

## `04_recencia_y_tuning.py`

Siete variantes sobre datos actuales, todas dentro de 0.17 puntos
(85.11% – 85.28%): ponderación por recencia (medias vidas de 7, 14 y 21
días), más iteraciones con learning rate menor, más hojas con
regularización, y calibración por estación estimada sobre histórico.

**Conclusión**: ninguna mejora; el modelo está en su techo con este enfoque.

## `05_calibracion_con_feedback.py`

Motivación: sobre 2.400 evaluaciones reales el modelo predice solo el 94.7%
de la demanda observada (0.825 en la estación 09122). El sesgo aparece en
todas las versiones del modelo, así que parece estructural: entrenar con
pérdida MAE apunta a la mediana, y con demanda sesgada a la derecha la
mediana queda por debajo de la media.

El backtest agregado (sin fuga, calibrando cada ciclo solo con evaluaciones
anteriores) daba **+0.33 puntos** con calibración por estación sobre los
últimos 12 ciclos.

**Pero el análisis pareado por ciclo lo desmiente**: mejora media de
**−0.559 puntos**, solo 23 de 45 ciclos mejoran (51%, una moneda al aire),
t = −1.38. El +0.33 era un artefacto de agrupar la métrica sobre todos los
ciclos a la vez, no una mejora real.

**Conclusión: descartada.** Queda documentada porque el resultado agregado
era engañoso y valía la pena dejar registrado por qué no se implementó.

## Nota sobre el margen con los punteros

La accuracy real por ciclo tiene media 82.73 y desviación 2.40 sobre 49
ciclos, así que el error estándar de la media es 0.34 puntos. La diferencia
con el primer lugar (0.49 puntos) son 1.4 errores estándar: estadísticamente
no se distingue de un empate.
