# Exploracion de episodios procesados

Solo fase RACE_PHASE. `off-road` = fraccion de frames con is_on_road=0.

| episodio | pista | dif | seg | pos | fin | v.media | accel>0.5 | |steer|>0.3 | brake | nitro | drift | fire | off-road |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| df1818d3 | gran_paradiso_island | 3 | 182 | 8 | True | 24.5 | 0.98 | 0.49 | 0.002 | 0.023 | 0.37 | 0.008 | 0.087 |
| e1279432 | lighthouse | 3 | 100 | 7 | False | 24.1 | 1.00 | 0.62 | 0.000 | 0.027 | 0.35 | 0.007 | 0.111 |
| edc8b8e2 | gran_paradiso_island | 3 | 177 | 6 | True | 24.9 | 0.98 | 0.55 | 0.009 | 0.025 | 0.44 | 0.006 | 0.134 |
| f5c0e900 | minigolf | 2 | 12 | - | - | sin fase de carrera | | | | | | | |

## Global (RACE_PHASE)

- steer: media +0.050, std 0.619, cuantiles 5/50/95 = [-1.0, 0.0, 1.0], valores unicos ~198
- acceleration: media 0.984, fraccion=1: 0.984
- brake: activo 0.43% de los frames
- nitro: activo 2.46% de los frames
- drift: activo 39.44% de los frames
- fire: activo 0.71% de los frames
- rescue: activo 0.00% de los frames
