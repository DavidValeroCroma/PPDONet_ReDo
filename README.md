# Replica PPDONet para mapas Planet2Disk

Este proyecto organiza una replica de modelos PPDONet para predecir mapas 2D de discos protoplanetarios a partir de parametros fisicos de simulacion y coordenadas espaciales. El codigo esta dividido en tres notebooks principales y un modulo comun reutilizable.

## Objetivo del proyecto

El objetivo es entrenar tres modelos PPDONet independientes para aproximar:

- `Sigma`: mapa de densidad superficial.
- `v_r`: mapa de velocidad radial.
- `v_theta`: mapa de velocidad azimutal.

Cada modelo aprende una funcion que toma como entrada:

- Parametros fisicos de simulacion:
  - `ALPHA`
  - `PLANETMASS`
  - `ASPECTRATIO`
- Identificacion espacial del valor del mapa:
  - `run`
  - `r`
  - `theta`

En la arquitectura PPDONet (variante de DeepONet), los parametros fisicos alimentan la red `branch`, mientras que las coordenadas espaciales `(r, theta)` alimentan la red `trunk`. El valor de salida corresponde al valor del campo fisico en una posicion del mapa.

## Estructura del proyecto

```text
Modelos/
+-- PPDONet-Stan-Sigma.ipynb
+-- PPDONet-Stan-Vr.ipynb
+-- PPDONet-Stan-Va.ipynb
+-- ppdonet_common.py
+-- README.md
```

### Notebooks

`PPDONet-Stan-Sigma.ipynb`

Entrena y evalua el modelo para densidad superficial. El target se transforma con `log10`, manteniendo la misma logica del codigo original.

`PPDONet-Stan-Vr.ipynb`

Entrena y evalua el modelo para velocidad radial. El target se procesa restando el fondo radial en la primera fila de la grilla.

`PPDONet-Stan-Va.ipynb`

Entrena y evalua el modelo para velocidad azimutal. La evaluacion y visualizacion usan el residual kepleriano:

```text
dv_theta = (v_theta + r) - sqrt(1/r)
```

Esto conserva el tratamiento especial aplicado al campo azimutal.

### Modulo comun

`ppdonet_common.py` concentra toda la logica repetida entre notebooks:

- descarga y apertura de datos;
- busqueda de parametros fisicos;
- construccion de coordenadas para trunk;
- definicion de la activacion STAN;
- definicion de MLP, branch, trunk y PPDONet;
- construccion de datasets y dataloaders;
- entrenamiento;
- guardado y carga de checkpoints;
- evaluacion;
- prediccion de un ejemplo;
- graficos 2D de verdad, prediccion y residuo.

## Dependencias principales

El proyecto usa:

```text
python
torch
numpy
xarray
huggingface_hub
tqdm
matplotlib
jupyter
```

Tambien se requiere un backend capaz de leer archivos NetCDF mediante `xarray`. Dependiendo del entorno, puede ser necesario instalar `netCDF4` o `h5netcdf`.

## Datos

Los datos se descargan automaticamente desde Hugging Face:

- `smao-astro/Planet2Disk_train`
- `smao-astro/Planet2Disk_test`

La funcion `download_planet2disk_data()` crea o reutiliza carpetas locales:

```text
Planet2Disk_train/
Planet2Disk_test/
```

Dentro de cada carpeta se esperan archivos NetCDF con los campos:

```text
batch_truth_sigma.nc
batch_truth_v_r.nc
batch_truth_v_theta.nc
```

## Arquitectura del modelo

El modelo implementado es una variante PPDONet compuesta por tres redes:

### Branch net

Recibe los parametros fisicos de cada simulacion:

```text
u = (ALPHA, ASPECTRATIO/h0, PLANETMASS/q)
```

En el codigo, estos parametros se cargan como un tensor de forma:

```text
(B, 3)
```

donde `B` es el numero de simulaciones.

### Trunk net

Recibe las coordenadas del mapa. La coordenada angular se codifica de forma periodica:

```text
(r, theta) -> (r_scaled, sin(theta), cos(theta))
```

Esto evita discontinuidades artificiales entre `theta = 0` y `theta = 2*pi`.

### Red de salida `z_net`

La salida de branch y trunk se combina mediante producto elemento a elemento. Luego `z_net` transforma esa representacion latente en un valor escalar del campo fisico:

```text
pred(run, r, theta)
```

## Activacion STAN

Los notebooks usan activacion `stan`, definida en la clase `Stan`.

La activacion aplicada es:

```text
tanh(x) * (1 + beta*x)
```

El parametro `beta` es entrenable. En la configuracion actual se usa:

```python
stan_beta = 0.1
stan_positive_beta = True
```

## Flujo de ejecucion

Cada notebook sigue el mismo flujo general.

### 1. Configuracion inicial

Se selecciona la GPU visible mediante:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # o "1"
```

Luego se importa todo desde `ppdonet_common.py` y se detecta el dispositivo:

```python
device = get_device()
print_device_info(device)
```

### 2. Descarga y apertura de datos

Se descargan los datasets de entrenamiento y prueba:

```python
train_dir, test_dir = download_planet2disk_data()
```

Luego se abren los tres campos:

```python
ds_sigma_tr, ds_vr_tr, ds_vtheta_tr = open_outputs(train_dir)
ds_sigma_te, ds_vr_te, ds_vtheta_te = open_outputs(test_dir)
```

### 3. Carga de parametros fisicos

Los parametros se buscan de manera robusta en archivos `.npy` o `.nc`:

```python
params_tr, where_tr = find_and_load_params(train_dir, device)
params_te, where_te = find_and_load_params(test_dir, device)
```

La salida esperada tiene forma:

```text
(B, 3)
```

### 4. Construccion de coordenadas

La funcion `build_coords()` crea la grilla completa `(r, theta)` y aplica la transformacion periodica:

```python
coords_tf, r_vals, th_vals = build_coords(ds_sigma_tr, device)
```

`coords_tf` queda con forma:

```text
(Ngrid, 3)
```

donde:

```text
Ngrid = Nr * Ntheta
```

### 5. Creacion de datasets y dataloaders

Cada notebook selecciona su campo:

```python
sigma_var = list(ds_sigma_tr.data_vars)[0]
vr_var = list(ds_vr_tr.data_vars)[0]
vtheta_var = list(ds_vtheta_tr.data_vars)[0]
```

Luego crea loaders con `make_field_loaders()`. Los flags cambian por modelo:

```text
Sigma:   log_target=True,  subtract_background=False
v_r:     log_target=False, subtract_background=True
v_theta: log_target=False, subtract_background=False
```

El valor:

```python
NPOINTS_SAMPLE = 435483
```

usa la grilla completa actual. Si se quiere entrenar con menos puntos por simulacion, este numero puede reducirse.

### 6. Entrenamiento

Cada notebook crea el modelo:

```python
model = build_ppdonet(act="stan", stan_beta=0.1, stan_positive_beta=True)
```

y lo entrena con:

```python
history, model = train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs=1500,
    lr=1e-4,
)
```

La funcion usa:

- optimizador `Adam`;
- perdida `MSELoss`;
- evaluacion de perdida en train y test al final de cada epoch.

### 7. Guardado del modelo

Al terminar el entrenamiento, cada notebook guarda un checkpoint:

```text
modelo_sigma_Stan.pt
modelo_vr_Stan.pt
modelo_vazimuth_Stan.pt
```

El guardado se hace con:

```python
save_model(model, CHECKPOINT)
```

### 8. Carga y evaluacion

Para evaluar sin reentrenar:

```python
model = load_model(CHECKPOINT, device)
```

Luego:

- `Sigma` y `v_r` usan `evaluate_model()`.
- `v_theta` usa `evaluate_model_vtheta_dvx()` para evaluar en el espacio residual kepleriano.

Las metricas reportadas son:

```text
MSE
R2
```

### 9. Prediccion de un caso de test

Cada notebook predice un caso individual:

```python
idx_example = 0
pred_map, true_map, mse_ex, r2_ex = predict_single_field(...)
```

La funcion devuelve:

- mapa predicho 2D;
- mapa verdadero 2D;
- MSE del ejemplo;
- R2 del ejemplo.

### 10. Visualizacion

La funcion `plot_field_maps()` muestra tres paneles:

```text
Ground Truth | Prediction | Residual (pred - true)
```

Para `v_theta`, se usa `robust=True`, lo que fija limites de color entre percentiles 1 y 99 para mejorar la visualizacion.

## Como ejecutar

Abrir Jupyter desde la carpeta del proyecto:

```bash
jupyter notebook
```

Luego ejecutar el notebook correspondiente:

```text
PPDONet-Stan-Sigma.ipynb
PPDONet-Stan-Vr.ipynb
PPDONet-Stan-Va.ipynb
```

Si ya existe el checkpoint correspondiente, se puede omitir la celda de entrenamiento y ejecutar directamente desde la seccion de evaluacion.

## Checkpoints esperados

Los notebooks usan estos nombres:

```text
modelo_sigma_Stan.pt
modelo_vr_Stan.pt
modelo_vazimuth_Stan.pt
```

Si el archivo no existe, primero se debe ejecutar la celda de entrenamiento y guardado.

## Notas de mantenimiento

- La logica principal del proyecto vive en `ppdonet_common.py`.
- Los notebooks quedan como scripts de ejecucion claros y cortos.
- Para cambiar la arquitectura, modificar `build_ppdonet()`.
- Para cambiar el preprocesamiento del target, modificar los flags del loader o `PPDONetDataset`.
- Para cambiar la metrica especial de `v_theta`, revisar `fargo_vtheta_delta()` y `evaluate_model_vtheta_dvx()`.
