# PPDONet con tiempo en el trunk network

Esta carpeta contiene una version espacio-temporal de los modelos PPDONet del proyecto principal. La diferencia central es que el tiempo se agrega como input esperado de la red `trunk`.

## Cambio principal respecto a la version base

En la version base, el trunk recibia:

```text
(r, theta) -> (r_scaled, sin(theta), cos(theta))
```

En esta version, el trunk recibe:

```text
(r, theta, time) -> (r_scaled, sin(theta), cos(theta), time_scaled)
```

Por lo tanto, la entrada del trunk pasa de dimension 3 a dimension 4. El branch se mantiene igual y sigue recibiendo los parametros fisicos de la simulacion:

```text
(ALPHA, ASPECTRATIO/h0, PLANETMASS/q)
```

## Estructura

```text
Models_time/
+-- PPDONet-Time-Stan-Sigma.ipynb
+-- PPDONet-Time-Stan-Vr.ipynb
+-- PPDONet-Time-Stan-Va.ipynb
+-- ppdonet_common.py
+-- README.md
```

## Archivos principales

`ppdonet_common.py`

Modulo comun de la version temporal. Incluye:

- descarga y apertura de datos;
- busqueda de parametros fisicos;
- deteccion de coordenada temporal;
- construccion de grilla `(time, r, theta)`;
- transformacion espacio-temporal para el trunk;
- arquitectura PPDONet con trunk de entrada 4;
- datasets y dataloaders compatibles con targets 3D o 4D;
- entrenamiento, evaluacion, guardado y carga;
- visualizacion de cortes temporales.

`PPDONet-Time-Stan-Sigma.ipynb`

Modelo temporal para densidad superficial. Usa `log10` sobre el target.

`PPDONet-Time-Stan-Vr.ipynb`

Modelo temporal para velocidad radial. Resta el fondo radial en cada tiempo.

`PPDONet-Time-Stan-Va.ipynb`

Modelo temporal para velocidad azimutal. Mantiene la evaluacion con residual kepleriano:

```text
dv_theta = (v_theta + r) - sqrt(1/r)
```

## Manejo del tiempo

La funcion `get_time_values()` busca una coordenada temporal con nombres comunes:

```text
time
t
snapshot
frame
output_time
```

Si el dataset no tiene una dimension temporal, el codigo usa:

```text
time = [0.0]
```

Esto permite que la version temporal tambien funcione con los datos estacionarios actuales. En ese caso, el trunk recibe tiempo, pero este no agrega variacion real porque solo existe un instante.

## Coordenadas del trunk

La funcion `build_coords()` crea una grilla completa:

```text
(time, r, theta)
```

Luego la aplana y transforma a:

```text
(r_scaled, sin(theta), cos(theta), time_scaled)
```

La forma final de `coords_tf` es:

```text
(Ntime * Nr * Ntheta, 4)
```

## Targets

La funcion `field_to_run_time_r_theta()` estandariza los campos de salida a:

```text
(run, time, r, theta)
```

Si el campo original tiene forma:

```text
(run, r, theta)
```

se agrega una dimension temporal artificial:

```text
(run, 1, r, theta)
```

Luego cada target se aplana para que coincida con la salida del modelo:

```text
(run, Ntime * Nr * Ntheta)
```

## Arquitectura

La arquitectura sigue el patron PPDONet:

### Branch

Entrada:

```text
(ALPHA, ASPECTRATIO/h0, PLANETMASS/q)
```

Salida:

```text
(B, d)
```

### Trunk temporal

Entrada:

```text
(r_scaled, sin(theta), cos(theta), time_scaled)
```

Salida:

```text
(Npoints, d)
```

### Combinacion

Branch y trunk se multiplican elemento a elemento:

```text
branch(u) * trunk(r, theta, time)
```

Luego `z_net` produce el valor escalar del campo:

```text
pred(run, time, r, theta)
```

## Flujo de ejecucion

Cada notebook sigue estos pasos:

1. Selecciona GPU y carga utilidades desde `ppdonet_common.py`.
2. Descarga o reutiliza los datasets Planet2Disk.
3. Abre los NetCDF de `sigma`, `v_r` y `v_theta`.
4. Carga parametros fisicos para branch.
5. Construye coordenadas espacio-temporales para trunk.
6. Crea datasets y dataloaders.
7. Entrena el modelo PPDONet temporal.
8. Guarda el checkpoint.
9. Carga el checkpoint.
10. Evalua MSE y R2.
11. Predice un ejemplo de test para un `time_index`.
12. Grafica ground truth, prediccion y residuo.

## Checkpoints

Los notebooks temporales guardan checkpoints con nombres distintos a la version base:

```text
modelo_sigma_time_Stan.pt
modelo_vr_time_Stan.pt
modelo_vazimuth_time_Stan.pt
```

Esto evita sobrescribir los pesos de los modelos sin tiempo.

## Ejecucion

Abrir Jupyter desde esta carpeta o desde la carpeta padre:

```bash
jupyter notebook
```

Ejecutar el notebook deseado:

```text
PPDONet-Time-Stan-Sigma.ipynb
PPDONet-Time-Stan-Vr.ipynb
PPDONet-Time-Stan-Va.ipynb
```

Si ya existe el checkpoint temporal correspondiente, se puede saltar la celda de entrenamiento y ejecutar directamente desde la seccion de evaluacion.

## Parametro NPOINTS_SAMPLE

En los notebooks se usa:

```python
NPOINTS_SAMPLE = None
```

Esto significa que se usan todos los puntos espacio-temporales. Para entrenamientos mas rapidos se puede cambiar por un entero, por ejemplo:

```python
NPOINTS_SAMPLE = 50000
```

En ese caso, cada simulacion entrega una muestra aleatoria de puntos `(time, r, theta)` en cada llamada del dataset.

## Notas importantes

- El branch no cambia respecto a la version original.
- El trunk ahora espera cuatro features.
- Si los datos actuales no incluyen tiempo, el codigo sigue siendo ejecutable con un unico tiempo artificial.
- Para datos realmente temporales, el archivo NetCDF debe incluir una dimension temporal reconocible por `get_time_values()`.
- La visualizacion muestra un corte temporal especificado por `time_index`.

