import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


#Entradas: ninguna.
#Salidas: string con el dispositivo disponible ("cuda" o "cpu").
# funcion que detecta si PyTorch puede usar GPU CUDA y, si no, selecciona CPU.
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


#Entradas: device, string con el dispositivo seleccionado.
#Salidas: ninguna; imprime informacion en pantalla.
# funcion que muestra la version de PyTorch, el dispositivo activo y el numero de GPUs disponibles.
def print_device_info(device: str) -> None:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"PyTorch: {torch.__version__}")
    print(f"Device : {device} ({gpu_name})")
    print(f"GPUs   : {torch.cuda.device_count()}")


#Entradas: base_dir, carpeta local donde se descargan los datasets.
#Salidas: rutas locales de los datasets de entrenamiento y test.
# funcion que descarga desde Hugging Face los datos Planet2Disk de entrenamiento y prueba.
def download_planet2disk_data(base_dir="."):
    train_dir = snapshot_download(
        repo_id="smao-astro/Planet2Disk_train",
        repo_type="dataset",
        local_dir=str(Path(base_dir) / "Planet2Disk_train"),
        local_dir_use_symlinks=False,
    )
    test_dir = snapshot_download(
        repo_id="smao-astro/Planet2Disk_test",
        repo_type="dataset",
        local_dir=str(Path(base_dir) / "Planet2Disk_test"),
        local_dir_use_symlinks=False,
    )
    return train_dir, test_dir


#Entradas: folder, carpeta que contiene los archivos NetCDF de salida.
#Salidas: tres xarray.Dataset para sigma, v_r y v_theta.
# funcion que abre los mapas de verdad asociados a densidad superficial, velocidad radial y velocidad azimutal.
def open_outputs(folder):
    sigma_path = os.path.join(folder, "batch_truth_sigma.nc")
    vr_path = os.path.join(folder, "batch_truth_v_r.nc")
    vtheta_path = os.path.join(folder, "batch_truth_v_theta.nc")
    return (
        xr.open_dataset(sigma_path),
        xr.open_dataset(vr_path),
        xr.open_dataset(vtheta_path),
    )


#Entradas: ds, xarray.Dataset donde se buscan parametros fisicos.
#Salidas: diccionario con parametros encontrados, por ejemplo alpha, h0, q o params_2d.
# funcion que inspecciona variables y coordenadas de un Dataset para localizar parametros de branch.
def find_params_in_dataset(ds):
    found = {}
    for var_name in ds.data_vars:
        name = var_name.lower()
        arr = ds[var_name].values
        if "alpha" in name and arr.ndim == 1:
            found["alpha"] = arr
        if ("h0" in name or "aspect" in name) and arr.ndim == 1:
            found["h0"] = arr
        if (name == "q" or "mass_ratio" in name or "planet" in name) and arr.ndim == 1:
            found["q"] = arr
        if arr.ndim == 2 and arr.shape[1] == 3:
            found["params_2d"] = arr

    for coord_name in ds.coords:
        name = coord_name.lower()
        arr = ds[coord_name].values
        if "alpha" in name and np.ndim(arr) == 1:
            found["alpha"] = arr
        if ("h0" in name or "aspect" in name) and np.ndim(arr) == 1:
            found["h0"] = arr
        if (name == "q" or "mass_ratio" in name or "planet" in name) and np.ndim(arr) == 1:
            found["q"] = arr

    return found


#Entradas: folder, carpeta del dataset; device, dispositivo donde se cargan los tensores.
#Salidas: tensor (B, 3) con parametros y string indicando desde donde fueron leidos.
# funcion que carga los parametros ALPHA, ASPECTRATIO/h0 y PLANETMASS/q desde archivos npy o NetCDF.
def find_and_load_params(folder, device):
    files = os.listdir(folder)
    for name in ["params.npy", "batch_params.npy", "batch_params_train.npy"]:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return torch.tensor(arr, dtype=torch.float32, device=device), f"params from {name}"

    alpha = h0 = q = None
    for name in [f for f in files if f.endswith(".nc")]:
        with xr.open_dataset(os.path.join(folder, name)) as ds:
            found = find_params_in_dataset(ds)

        if "params_2d" in found:
            params = torch.tensor(found["params_2d"], dtype=torch.float32, device=device)
            return params, f"params_2d from {name}"
        if "alpha" in found and alpha is None:
            alpha = found["alpha"]
        if "h0" in found and h0 is None:
            h0 = found["h0"]
        if "q" in found and q is None:
            q = found["q"]

    if alpha is not None and h0 is not None and q is not None:
        params = np.stack([alpha, h0, q], axis=1)
        return torch.tensor(params, dtype=torch.float32, device=device), "params from alpha/h0/q in nc"

    patterns = ["*param*", "*alpha*", "*h0*", "*aspect*", "*q*"]
    hits = []
    for pattern in patterns:
        hits.extend(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    raise FileNotFoundError(
        "No se encontraron parametros alpha, h0/aspect ratio y q/planet mass.\n"
        "Archivos candidatos:\n" + "\n".join(sorted(set(hits)))
    )


#Entradas: u, valor o tensor a escalar; u_min y u_max, limites del intervalo original.
#Salidas: valor o tensor reescalado al intervalo aproximado [-1, 1].
# funcion que normaliza coordenadas continuas para alimentar de forma estable la red trunk.
def scale_to_one(u, u_min, u_max):
    if float(u_max) == float(u_min):
        return torch.zeros_like(u)
    midpoint = (u_min + u_max) / 2.0
    return (u - midpoint) / (u_max - u_min) * 2.0


#Entradas: ds_reference, Dataset con una posible coordenada temporal.
#Salidas: array 1D con valores de tiempo; si no existe tiempo, retorna [0.0].
# funcion que extrae una coordenada temporal compatible con nombres usuales o crea un tiempo estacionario.
def get_time_values(ds_reference):
    time_candidates = ["time", "t", "snapshot", "frame", "output_time"]
    for name in time_candidates:
        if name in ds_reference.coords:
            return np.asarray(ds_reference[name].values, dtype=np.float32)
        if name in ds_reference.dims:
            return np.arange(ds_reference.sizes[name], dtype=np.float32)
    return np.array([0.0], dtype=np.float32)


#Entradas: r_min, r_max, t_min y t_max, limites radial y temporal.
#Salidas: funcion transformadora que convierte (r, theta, time) en (r_scaled, sin(theta), cos(theta), t_scaled).
# funcion que construye la transformacion espacio-temporal usada por la red trunk.
def space_time_transform(r_min, r_max, t_min, t_max):
    #Entradas: y, tensor de coordenadas con columnas (r, theta, time).
    #Salidas: tensor con columnas (r_scaled, sin(theta), cos(theta), t_scaled).
    # funcion que aplica escalamiento radial/temporal y codificacion periodica angular.
    def transform(y):
        r = y[..., 0:1]
        theta = y[..., 1:2]
        time = y[..., 2:3]
        r_scaled = scale_to_one(r, r_min, r_max)
        t_scaled = scale_to_one(time, t_min, t_max)
        return torch.cat([r_scaled, torch.sin(theta), torch.cos(theta), t_scaled], dim=-1)

    return transform


#Entradas: ds_reference, Dataset con coordenadas r/theta/time; device, dispositivo destino.
#Salidas: coords_tf, r_vals, theta_vals y time_vals.
# funcion que construye la grilla completa espacio-temporal para alimentar el trunk network.
def build_coords(ds_reference, device):
    r_vals = ds_reference["r"].values
    theta_vals = ds_reference["theta"].values
    time_vals = get_time_values(ds_reference)
    time_grid, r_grid, theta_grid = torch.meshgrid(
        torch.tensor(time_vals, dtype=torch.float32),
        torch.tensor(r_vals, dtype=torch.float32),
        torch.tensor(theta_vals, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack(
        [r_grid.reshape(-1), theta_grid.reshape(-1), time_grid.reshape(-1)],
        dim=-1,
    )
    coords_tf = space_time_transform(
        r_vals.min(),
        r_vals.max(),
        time_vals.min(),
        time_vals.max(),
    )(coords).to(device)
    return coords_tf, r_vals, theta_vals, time_vals


#Entradas: data_array, xarray.DataArray del campo objetivo; time_vals, tiempos esperados.
#Salidas: array numpy ordenado como (run, time, r, theta).
# funcion que estandariza el orden de dimensiones del target para hacerlo compatible con coords_tf.
def field_to_run_time_r_theta(data_array, time_vals):
    dims = list(data_array.dims)
    r_dim = next((dim for dim in dims if dim.lower() == "r"), None)
    theta_dim = next((dim for dim in dims if "theta" in dim.lower()), None)
    time_dim = next(
        (dim for dim in dims if dim.lower() in {"time", "t", "snapshot", "frame", "output_time"}),
        None,
    )
    run_dim = next((dim for dim in dims if dim not in {r_dim, theta_dim, time_dim}), None)

    if r_dim is None or theta_dim is None or run_dim is None:
        raise ValueError(f"No se pudieron identificar dimensiones run/r/theta en {dims}")

    if time_dim is None:
        arr = data_array.transpose(run_dim, r_dim, theta_dim).values.astype(np.float32)
        return arr[:, None, :, :]

    arr = data_array.transpose(run_dim, time_dim, r_dim, theta_dim).values.astype(np.float32)
    if arr.shape[1] != len(time_vals):
        raise ValueError("La dimension temporal del campo no coincide con time_vals.")
    return arr


class Stan(nn.Module):
    #Entradas: beta inicial y positive_beta para restringir beta a valores positivos.
    #Salidas: instancia de la activacion STAN.
    # funcion que inicializa la activacion STAN entrenable usada en las capas MLP.
    def __init__(self, beta: float = 0.1, positive_beta: bool = False):
        super().__init__()
        self._beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))
        self.positive_beta = bool(positive_beta)

    #Entradas: x, tensor de activaciones de una capa lineal.
    #Salidas: tensor transformado por tanh(x) * (1 + beta*x).
    # funcion que aplica la activacion STAN durante el paso forward.
    def forward(self, x):
        beta = F.softplus(self._beta) if self.positive_beta else self._beta
        return torch.tanh(x) * (1.0 + beta * x)


class MLP(nn.Module):
    #Entradas: dimensiones de entrada/salida, tamanos ocultos y configuracion de activacion.
    #Salidas: instancia de una red MLP inicializada.
    # funcion que construye una red completamente conectada con activacion configurable.
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: list[int],
        activation="tanh",
        stan_beta: float = 0.1,
        stan_positive_beta: bool = False,
    ):
        super().__init__()
        if isinstance(activation, nn.Module):
            self.act = activation
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "stan":
            self.act = Stan(beta=stan_beta, positive_beta=stan_positive_beta)
        else:
            raise ValueError(f"Activacion no soportada: {activation}")

        sizes = [in_dim] + list(hidden_sizes) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    #Entradas: x, tensor de entrada para la red MLP.
    #Salidas: tensor de salida luego de aplicar capas lineales y activaciones.
    # funcion que ejecuta el paso forward de la MLP.
    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return self.layers[-1](h)


class TriDeepONet(nn.Module):
    #Entradas: branch_net, trunk_net y z_net ya construidas.
    #Salidas: instancia de TriDeepONet.
    # funcion que inicializa la arquitectura PPDONet con redes branch, trunk y red de combinacion.
    def __init__(self, branch_net, trunk_net, z_net):
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.z_net = z_net

    #Entradas: u, parametros fisicos del branch; y, coordenadas espacio-temporales transformadas del trunk.
    #Salidas: prediccion del campo para cada simulacion y punto espacio-temporal.
    # funcion que combina branch y trunk por producto elemento a elemento y produce el mapa final.
    def forward(self, u, y):
        branch = self.branch_net(u).unsqueeze(1)
        trunk = self.trunk_net(y)
        if trunk.dim() == 2:
            trunk = trunk.unsqueeze(0)
        elif trunk.dim() != 3:
            raise ValueError(f"trunk_net(y) retorno una forma invalida: {trunk.shape}")

        out = self.z_net(branch * trunk)
        return out.squeeze(-1)


#Entradas: activacion, parametros STAN y dimension latente d.
#Salidas: modelo TriDeepONet configurado para PPDONet con trunk espacio-temporal.
# funcion que construye el modelo completo con branch de parametros y trunk de dimension 4: r, sin(theta), cos(theta), tiempo.
def build_ppdonet(
    act="stan",
    stan_beta: float = 0.1,
    stan_positive_beta: bool = True,
    d: int = 50,
):
    branch = MLP(
        3,
        d,
        hidden_sizes=[100, 100, 100, 100],
        activation=act,
        stan_beta=stan_beta,
        stan_positive_beta=stan_positive_beta,
    )
    trunk = MLP(
        4,
        d,
        hidden_sizes=[256, 256, 256, 256, 256],
        activation=act,
        stan_beta=stan_beta,
        stan_positive_beta=stan_positive_beta,
    )
    z_net = MLP(
        d,
        1,
        hidden_sizes=[100],
        activation=act,
        stan_beta=stan_beta,
        stan_positive_beta=stan_positive_beta,
    )
    return TriDeepONet(branch, trunk, z_net)


class PPDONetDataset(Dataset):
    #Entradas: Dataset xarray, parametros, coordenadas espacio-temporales, nombre del campo y opciones de preprocesamiento.
    #Salidas: instancia Dataset lista para DataLoader.
    # funcion que prepara pares (parametros, coordenadas espacio-temporales, target) para entrenar o evaluar PPDONet.
    def __init__(
        self,
        ds_out,
        params,
        coords_tf,
        field_name,
        time_vals,
        device,
        log_target=False,
        subtract_background=False,
        n_points_sample=None,
        seed=0,
    ):
        self.coords = coords_tf.to(device).float()
        self.params = params.to(device).float()
        self.n_points_sample = n_points_sample
        self.rng = np.random.default_rng(seed)
        self.device = device

        out_np = field_to_run_time_r_theta(ds_out[field_name], time_vals)
        out = torch.tensor(out_np, dtype=torch.float32, device=device)
        if subtract_background:
            out = out - out[:, :, 0:1, :]
        if log_target:
            out = torch.log10(torch.clamp(out, min=1e-12))

        batch_size, n_time, n_r, n_theta = out.shape
        self.targets_full = out.reshape(batch_size, n_time * n_r * n_theta)

    #Entradas: ninguna.
    #Salidas: numero de simulaciones disponibles.
    # funcion que informa el largo del dataset en cantidad de simulaciones.
    def __len__(self):
        return self.params.shape[0]

    #Entradas: idx, indice de la simulacion solicitada.
    #Salidas: u, y y target; opcionalmente sampleados en una subgrilla espacio-temporal.
    # funcion que entrega un ejemplo compuesto por parametros branch, coordenadas trunk y valores objetivo.
    def __getitem__(self, idx):
        u = self.params[idx]
        y = self.coords
        target = self.targets_full[idx]
        if self.n_points_sample is None:
            return u, y, target

        n_sample = min(self.n_points_sample, len(y))
        ids = self.rng.choice(len(y), size=n_sample, replace=False)
        ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        return u, y[ids], target[ids]


#Entradas: datasets train/test, parametros, coordenadas, tiempos, campo objetivo y opciones de preprocesamiento.
#Salidas: DataLoader de entrenamiento y DataLoader de test.
# funcion que crea los loaders para un campo especifico manteniendo la configuracion comun del proyecto temporal.
def make_field_loaders(
    ds_train,
    ds_test,
    params_train,
    params_test,
    coords_tf,
    field_name,
    time_vals,
    device,
    log_target=False,
    subtract_background=False,
    n_points_sample=None,
    batch_size=1,
):
    train_ds = PPDONetDataset(
        ds_train,
        params_train,
        coords_tf,
        field_name,
        time_vals,
        device,
        log_target=log_target,
        subtract_background=subtract_background,
        n_points_sample=n_points_sample,
    )
    test_ds = PPDONetDataset(
        ds_test,
        params_test,
        coords_tf,
        field_name,
        time_vals,
        device,
        log_target=log_target,
        subtract_background=subtract_background,
        n_points_sample=n_points_sample,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


#Entradas: modelo, loaders, dispositivo, numero de epochs y learning rate.
#Salidas: historial de perdidas y modelo entrenado.
# funcion que entrena PPDONet con MSE, Adam y evaluacion de perdida en test por epoch.
def train_model(model, train_loader, test_loader, device, epochs=1500, lr=1e-4):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for u, y, target in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            pred = model(u, y)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        test_losses = []
        with torch.no_grad():
            for u, y, target in test_loader:
                test_losses.append(loss_fn(model(u, y), target).item())

        train_mean = float(np.mean(train_losses))
        test_mean = float(np.mean(test_losses))
        history["train_loss"].append(train_mean)
        history["test_loss"].append(test_mean)
        print(f"Epoch {epoch}: train={train_mean:.4e} test={test_mean:.4e}")

    return history, model


#Entradas: modelo entrenado y ruta de archivo.
#Salidas: ninguna; escribe el state_dict en disco.
# funcion que guarda los pesos entrenados del modelo en un checkpoint .pt.
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Parametros guardados en {path}")


#Entradas: ruta del checkpoint y dispositivo destino.
#Salidas: modelo PPDONet cargado con los pesos guardados.
# funcion que reconstruye la arquitectura temporal y carga sus parametros entrenados desde disco.
def load_model(path, device):
    model = build_ppdonet(act="stan", stan_beta=0.1, stan_positive_beta=True)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


#Entradas: pred, tensor predicho; target, tensor verdadero.
#Salidas: valor R2 entre prediccion y verdad.
# funcion que calcula el coeficiente de determinacion para medir calidad de ajuste.
def r2_score(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1.0 - ss_res / ss_tot


#Entradas: modelo, loader de evaluacion y dispositivo.
#Salidas: MSE promedio y R2 promedio.
# funcion que evalua un modelo en el mismo espacio de target usado durante entrenamiento.
def evaluate_model(model, loader, device):
    model.eval()
    model.to(device)
    mses, r2s = [], []
    with torch.no_grad():
        for u, y, target in loader:
            pred = model(u, y).squeeze(0)
            target = target.squeeze(0)
            mses.append(F.mse_loss(pred, target).item())
            r2s.append(r2_score(pred, target).item())
    return float(np.mean(mses)), float(np.mean(r2s))


#Entradas: map_2d de v_theta, radios r_vals y epsilon numerico.
#Salidas: mapa residual dv_theta = (v_theta + r) - sqrt(1/r).
# funcion que convierte la velocidad azimutal almacenada por FARGO al residual kepleriano fisico.
def fargo_vtheta_delta(map_2d, r_vals, eps=1e-12):
    n_r, _ = map_2d.shape
    r = torch.tensor(r_vals, dtype=torch.float32, device=map_2d.device).view(n_r, 1)
    return (map_2d + r) - torch.sqrt(1.0 / torch.clamp(r, min=eps))


#Entradas: modelo de v_theta, loader, radios r_vals y dispositivo.
#Salidas: MSE promedio y R2 promedio en el espacio residual dv_theta.
# funcion que evalua velocidad azimutal tras convertir prediccion y target a delta kepleriano usando la columna radial del trunk.
def evaluate_model_vtheta_dvx(model, loader, r_vals, device):
    model.eval()
    model.to(device)
    mses, r2s = [], []
    r_min = float(np.min(r_vals))
    r_max = float(np.max(r_vals))
    r_mid = (r_min + r_max) / 2.0

    with torch.no_grad():
        for u, y, target in loader:
            pred = model(u, y).squeeze(0)
            target = target.squeeze(0)
            y_points = y.squeeze(0) if y.dim() == 3 and y.shape[0] == 1 else y
            r_real = y_points[:, 0] * (r_max - r_min) / 2.0 + r_mid
            kepler = torch.sqrt(1.0 / torch.clamp(r_real, min=1e-12))
            pred_dvx = (pred + r_real) - kepler
            true_dvx = (target + r_real) - kepler
            mses.append(F.mse_loss(pred_dvx, true_dvx).item())
            r2s.append(r2_score(pred_dvx, true_dvx).item())
    return float(np.mean(mses)), float(np.mean(r2s))


#Entradas: modelo, parametros, coordenadas, Dataset del campo, nombre del campo, grillas y opciones de transformacion.
#Salidas: mapa predicho, mapa verdadero, MSE y R2 para una simulacion y tiempo especificos.
# funcion que predice un caso individual de test para un indice temporal seleccionado.
def predict_single_field(
    model,
    params_all,
    coords_tf,
    ds_field,
    field_name,
    r_vals,
    theta_vals,
    time_vals,
    device,
    idx_example=0,
    time_index=0,
    log_target=False,
    subtract_background=False,
    fargo_kepler_delta=False,
):
    model.eval()
    model.to(device)
    n_time = len(time_vals)
    n_r = len(r_vals)
    n_theta = len(theta_vals)

    u = params_all[idx_example : idx_example + 1].to(device)
    with torch.no_grad():
        pred = model(u, coords_tf.to(device)).squeeze(0).cpu()

    true_all = field_to_run_time_r_theta(ds_field[field_name], time_vals)
    true_map = torch.tensor(true_all[idx_example], dtype=torch.float32)
    pred_map = pred.reshape(n_time, n_r, n_theta)

    if subtract_background:
        true_map = true_map - true_map[:, 0:1, :]
    if log_target:
        true_map = torch.log10(torch.clamp(true_map, min=1e-12))

    true_slice = true_map[time_index]
    pred_slice = pred_map[time_index]
    if fargo_kepler_delta:
        true_slice = fargo_vtheta_delta(true_slice, r_vals)
        pred_slice = fargo_vtheta_delta(pred_slice, r_vals)

    true_flat = true_slice.reshape(-1)
    pred_flat = pred_slice.reshape(-1)
    mse = F.mse_loss(pred_flat, true_flat).item()
    r2 = r2_score(pred_flat, true_flat).item()
    return pred_slice, true_slice, mse, r2


#Entradas: mapa predicho, mapa verdadero, grilla r/theta, titulo y opciones graficas.
#Salidas: ninguna; muestra figura con verdad, prediccion y residuo.
# funcion que grafica comparativamente los mapas 2D de ground truth, prediccion y residuo.
def plot_field_maps(pred_map, true_map, r_vals, theta_vals, title, cmap="viridis", robust=False):
    true_np = true_map.cpu().numpy() if hasattr(true_map, "cpu") else true_map
    pred_np = pred_map.cpu().numpy() if hasattr(pred_map, "cpu") else pred_map
    resid_np = pred_np - true_np
    extent = [theta_vals.min(), theta_vals.max(), r_vals.min(), r_vals.max()]
    color_limits = {}
    if robust:
        color_limits = dict(zip(["vmin", "vmax"], np.percentile(true_np, [1, 99])))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=14)

    im0 = axes[0].imshow(true_np, origin="lower", aspect="auto", extent=extent, cmap=cmap, **color_limits)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel(r"$r$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_np, origin="lower", aspect="auto", extent=extent, cmap=cmap, **color_limits)
    axes[1].set_title("Prediction")
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel(r"$r$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(resid_np, origin="lower", aspect="auto", extent=extent, cmap="seismic")
    axes[2].set_title("Residual (pred - true)")
    axes[2].set_xlabel(r"$\theta$")
    axes[2].set_ylabel(r"$r$")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
