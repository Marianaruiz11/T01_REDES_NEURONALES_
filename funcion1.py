"""
Trabajo 1 - Parte 1: optimización numérica sobre Rosenbrock (2D y 3D).

Métodos: descenso por gradiente (GD), algoritmo genético (GA), PSO y evolución
diferencial (DE). Además dos líneas base opcionales: búsqueda aleatoria y
recocido simulado.

Comandos (todos con un solo comando, sin editar código):
    python funcion1.py check
    python funcion1.py run   --algo pso --dim 2 --seed 42
    python funcion1.py stats --runs 30 --dims 2 3
    python funcion1.py gif   --algo gradient_descent --out figs/gd.gif

Los hiperparámetros se cambian con --param (p. ej. --param lr0=0.5
--param pso.w=0.6) o con un archivo --config config.json.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


# ======================================================================
# 1. Función de prueba
# ======================================================================
def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rosenbrock_grad(x):
    """Gradiente analítico. Solo se usa en `check` para validar el numérico."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return g


PROBLEMS: dict[str, dict] = {}


def register_problem(name, func, bounds, supported_dims=None,
                     known_optimum=None, grad=None):
    PROBLEMS[name] = {"func": func, "bounds": bounds,
                      "supported_dims": supported_dims,
                      "known_optimum": known_optimum, "grad": grad}


register_problem(
    "rosenbrock", rosenbrock, bounds=(-2.048, 2.048), supported_dims=(2, 3),
    known_optimum=lambda d: np.ones(d), grad=rosenbrock_grad,
)


# ======================================================================
# 2. Conteo de evaluaciones (f y ∇f por separado)
# ======================================================================
class CountedObjective:
    """
    n_f     : llamadas directas a f (población, búsqueda lineal, punto inicial).
    n_grad  : número de gradientes calculados.
    Equivalencia: un gradiente por diferencias centrales cuesta 2*dim
    evaluaciones de f, así que  n_equiv = n_f + 2*dim*n_grad.
    Las 2*dim evaluaciones internas del gradiente NO se suman a n_f
    (se cuentan una sola vez, dentro de n_grad).
    """

    def __init__(self, func: Callable, dim: int):
        self._func = func
        self.dim = dim
        self.n_f = 0
        self.n_grad = 0
        self.raw_calls = 0      # llamadas reales a func; solo para verificar

    @property
    def cost_grad(self) -> int:
        return 2 * self.dim

    @property
    def n_equiv(self) -> int:
        return self.n_f + self.cost_grad * self.n_grad

    def _raw(self, x) -> float:
        self.raw_calls += 1
        return float(self._func(x))

    def __call__(self, x) -> float:
        self.n_f += 1
        return self._raw(x)

    def gradient(self, x, h: float = 1e-6) -> np.ndarray:
        self.n_grad += 1
        g = np.empty(self.dim)
        for i in range(self.dim):
            e = np.zeros(self.dim)
            e[i] = h
            g[i] = (self._raw(x + e) - self._raw(x - e)) / (2.0 * h)
        return g


# ======================================================================
# 3. Resultado
# ======================================================================
@dataclass
class OptimizeResult:
    problem: str
    algorithm: str
    dim: int
    random_seed: int | None
    x_best: np.ndarray
    f_best: float
    n_f: int
    n_grad: int
    n_equiv: int
    history: list = field(default_factory=list, repr=False)     # mejor f por iteración
    eval_trace: list = field(default_factory=list, repr=False)  # n_equiv en cada iteración
    frames: list | None = field(default=None, repr=False)       # para GIF

    def __str__(self) -> str:
        return (
            f"{self.algorithm} | {self.problem} {self.dim}D | seed={self.random_seed}\n"
            f"  x_best  = {np.array2string(self.x_best, precision=6)}\n"
            f"  f_best  = {self.f_best:.6e}\n"
            f"  evals f = {self.n_f} | evals grad = {self.n_grad} "
            f"| equivalentes (f) = {self.n_equiv}"
        )


# ======================================================================
# 4. Algoritmos
# ======================================================================
# Firma común:
#   algo(obj, dim, bounds, rng, max_evals, record, **params)
#       -> (x_best, f_best, history, eval_trace, frames)
# `max_evals` es un presupuesto en evaluaciones EQUIVALENTES de f (obj.n_equiv),
# idéntico para todos los métodos => comparación justa.
ALGORITHMS: dict[str, Callable] = {}
_RESERVED = {"obj", "dim", "bounds", "rng", "max_evals", "record"}


def register_algorithm(name: str):
    def decorator(fn):
        ALGORITHMS[name] = fn
        return fn
    return decorator


def accepted_params(algo: str) -> set[str]:
    return set(inspect.signature(ALGORITHMS[algo]).parameters) - _RESERVED


# ---------------- Descenso por gradiente ----------------
@register_algorithm("gradient_descent")
def gradient_descent(obj, dim, bounds, rng, max_evals, record=False,
                     tol=1e-8, h=1e-6, lr0=1.0, c=1e-4, shrink=0.5, max_ls=60, x0=None):
    """x_{k+1} = P(x_k - a_k g_k), g_k por diferencias centrales, a_k por
    Armijo con retroceso (a <- shrink*a) y P = proyección al dominio."""
    lo, hi = bounds
    if not (np.isfinite([tol,h,lr0,c,shrink]).all() and tol >= 0 and h > 0
            and lr0 > 0 and 0 < c < 1 and 0 < shrink < 1
            and isinstance(max_ls, int) and max_ls >= 1):
        raise ValueError('Revise tol, h, lr0, c, shrink y max_ls')
    if x0 is None:
        x = rng.uniform(lo, hi, dim)
    else:
        x = np.asarray(x0, dtype=float)
        if x.shape != (dim,) or not np.isfinite(x).all():
            raise ValueError(f'x0 debe contener {dim} coordenadas finitas')
        if ((x < lo) | (x > hi)).any():
            raise ValueError(f'x0 debe estar dentro de [{lo}, {hi}]; no se recorta')
        x = x.copy()
    fx = obj(x)
    hist, ev = [fx], [obj.n_equiv]
    frames = [x.copy()] if record else None

    while obj.n_equiv + obj.cost_grad + 1 <= max_evals:
        g = obj.gradient(x, h)
        if np.linalg.norm(g) < tol:
            break
        lr, accepted = lr0, False
        for _ in range(max_ls):
            if obj.n_equiv + 1 > max_evals:
                break
            cand = np.clip(x - lr * g, lo, hi)
            fc = obj(cand)
            # Armijo con proyección: f(cand) <= f(x) - c * g.(x - cand)
            if fc <= fx - c * float(g @ (x - cand)):
                accepted = True
                break
            lr *= shrink
        if not accepted or np.array_equal(cand, x):
            break
        x, fx = cand, fc
        hist.append(fx)
        ev.append(obj.n_equiv)
        if record:
            frames.append(x.copy())
    return x, fx, hist, ev, frames


# ---------------- Algoritmo genético (codificación real) ----------------
@register_algorithm("genetic")
def genetic(obj, dim, bounds, rng, max_evals, record=False,
            pop_size=50, elite=2, tournament=3, pc=0.9, blx_alpha=0.5,
            sigma0=0.1, sigma_end=1e-4):
    """Selección por torneo, cruce BLX-alpha (prob. pc), mutación gaussiana por
    gen (prob. 1/dim) con sigma decreciente (geométrico, fracción del rango) y
    elitismo."""
    lo, hi = bounds
    rango = hi - lo
    n_off = pop_size - elite
    pop = rng.uniform(lo, hi, (pop_size, dim))
    fit = np.array([obj(p) for p in pop])
    hist, ev = [float(fit.min())], [obj.n_equiv]
    frames = [pop[:, :2].copy()] if record else None

    while obj.n_equiv + n_off <= max_evals:
        sigma = sigma0 * rango * (sigma_end / sigma0) ** (obj.n_equiv / max_evals)
        # torneo: para cada hijo, dos padres
        idx = rng.integers(0, pop_size, (n_off, 2, tournament))
        best_in_t = fit[idx].argmin(axis=2)
        win = np.take_along_axis(idx, best_in_t[..., None], axis=2)[..., 0]
        p1, p2 = pop[win[:, 0]], pop[win[:, 1]]
        # BLX-alpha
        d = np.abs(p1 - p2)
        child = rng.uniform(np.minimum(p1, p2) - blx_alpha * d,
                            np.maximum(p1, p2) + blx_alpha * d)
        child = np.where((rng.random(n_off) < pc)[:, None], child, p1)
        # mutación gaussiana
        mut = rng.random((n_off, dim)) < 1.0 / dim
        child = np.clip(child + mut * rng.normal(0.0, sigma, (n_off, dim)), lo, hi)
        cfit = np.array([obj(c) for c in child])
        # elitismo
        order = np.argsort(fit)[:elite]
        pop = np.vstack([pop[order], child])
        fit = np.concatenate([fit[order], cfit])
        hist.append(float(fit.min()))
        ev.append(obj.n_equiv)
        if record:
            frames.append(pop[:, :2].copy())
    i = int(fit.argmin())
    return pop[i].copy(), float(fit[i]), hist, ev, frames


# ---------------- PSO ----------------
@register_algorithm("pso")
def pso(obj, dim, bounds, rng, max_evals, record=False,
        n_particles=30, w=0.7298, c1=1.49618, c2=1.49618, vmax_frac=0.2):
    """v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x);  x <- x + v.
    Topología global (gbest). Al tocar la frontera: x se recorta y v=0."""
    lo, hi = bounds
    vmax = vmax_frac * (hi - lo)
    x = rng.uniform(lo, hi, (n_particles, dim))
    v = rng.uniform(-vmax, vmax, (n_particles, dim))
    fx = np.array([obj(p) for p in x])
    pbest, pfit = x.copy(), fx.copy()
    g = int(pfit.argmin())
    gbest, gfit = pbest[g].copy(), float(pfit[g])
    hist, ev = [gfit], [obj.n_equiv]
    frames = [x[:, :2].copy()] if record else None

    while obj.n_equiv + n_particles <= max_evals:
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        v = np.clip(w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x), -vmax, vmax)
        x = x + v
        out = (x < lo) | (x > hi)
        x = np.clip(x, lo, hi)
        v[out] = 0.0
        fx = np.array([obj(p) for p in x])
        better = fx < pfit
        pbest[better], pfit[better] = x[better], fx[better]
        g = int(pfit.argmin())
        if pfit[g] < gfit:
            gbest, gfit = pbest[g].copy(), float(pfit[g])
        hist.append(gfit)
        ev.append(obj.n_equiv)
        if record:
            frames.append(x[:, :2].copy())
    return gbest, gfit, hist, ev, frames


# ---------------- Evolución diferencial ----------------
@register_algorithm("differential_evolution")
def differential_evolution(obj, dim, bounds, rng, max_evals, record=False,
                           pop_size=40, F=0.5, CR=0.9):
    """DE/rand/1/bin:  v = x_a + F (x_b - x_c);  cruce binomial con CR (al
    menos un gen del mutante);  selección voraz  u reemplaza a x si f(u)<=f(x)."""
    if pop_size < 4:
        raise ValueError("DE requiere pop_size >= 4")
    lo, hi = bounds
    pop = rng.uniform(lo, hi, (pop_size, dim))
    fit = np.array([obj(p) for p in pop])
    hist, ev = [float(fit.min())], [obj.n_equiv]
    frames = [pop[:, :2].copy()] if record else None
    rows = np.arange(pop_size)

    while obj.n_equiv + pop_size <= max_evals:
        # tres índices distintos entre sí y distintos de i
        r = rng.random((pop_size, pop_size))
        r[rows, rows] = 2.0
        a, b, c = np.argsort(r, axis=1)[:, :3].T
        mutant = pop[a] + F * (pop[b] - pop[c])
        mask = rng.random((pop_size, dim)) < CR
        mask[rows, rng.integers(0, dim, pop_size)] = True
        trial = np.where(mask, mutant, pop)
        out = (trial < lo) | (trial > hi)
        trial = np.where(out, rng.uniform(lo, hi, trial.shape), trial)
        tfit = np.array([obj(u) for u in trial])
        win = tfit <= fit
        pop[win], fit[win] = trial[win], tfit[win]
        hist.append(float(fit.min()))
        ev.append(obj.n_equiv)
        if record:
            frames.append(pop[:, :2].copy())
    i = int(fit.argmin())
    return pop[i].copy(), float(fit[i]), hist, ev, frames


# ---------------- Líneas base (opcionales) ----------------
@register_algorithm("random_search")
def random_search(obj, dim, bounds, rng, max_evals, record=False):
    lo, hi = bounds
    x_best = rng.uniform(lo, hi, dim)
    f_best = obj(x_best)
    hist, ev = [f_best], [obj.n_equiv]
    while obj.n_equiv + 1 <= max_evals:
        x = rng.uniform(lo, hi, dim)
        fx = obj(x)
        if fx < f_best:
            x_best, f_best = x, fx
        hist.append(f_best)
        ev.append(obj.n_equiv)
    return x_best, f_best, hist, ev, None


@register_algorithm("simulated_annealing")
def simulated_annealing(obj, dim, bounds, rng, max_evals, record=False,
                        T0=10.0, alpha=0.9995, step=0.1):
    lo, hi = bounds
    x = rng.uniform(lo, hi, dim)
    fx = obj(x)
    x_best, f_best = x.copy(), fx
    hist, ev = [f_best], [obj.n_equiv]
    frames = [x.copy()] if record else None
    T = T0
    while obj.n_equiv + 1 <= max_evals:
        cand = np.clip(x + rng.normal(0.0, step, dim), lo, hi)
        fc = obj(cand)
        df = fc - fx
        if df < 0 or rng.random() < np.exp(-df / max(T, 1e-12)):
            x, fx = cand, fc
            if fx < f_best:
                x_best, f_best = x.copy(), fx
        T *= alpha
        hist.append(f_best)
        ev.append(obj.n_equiv)
        if record:
            frames.append(x.copy())
    return x_best, f_best, hist, ev, frames


MAIN_ALGOS = ("gradient_descent", "genetic", "pso", "differential_evolution")


# ======================================================================
# 5. Punto de entrada
# ======================================================================
def optimize(algorithm="gradient_descent", problem="rosenbrock", dim=2,
             random_seed=42, max_evals=20000, record=False,
             **algo_params) -> OptimizeResult:
    if not isinstance(max_evals, int) or max_evals < 1:
        raise ValueError('max_evals debe ser un entero positivo')
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo '{algorithm}' no registrado. Opciones: {sorted(ALGORITHMS)}")
    if problem not in PROBLEMS:
        raise ValueError(f"Problema '{problem}' no registrado. Opciones: {sorted(PROBLEMS)}")
    spec = PROBLEMS[problem]
    allowed = spec["supported_dims"]
    if allowed is not None and dim not in allowed:
        raise ValueError(f"'{problem}' admite dim en {allowed}, se recibió {dim}.")
    bad = set(algo_params) - accepted_params(algorithm)
    if bad:
        raise ValueError(f"Parámetros inválidos para '{algorithm}': {sorted(bad)}. "
                         f"Válidos: {sorted(accepted_params(algorithm))}")

    rng = np.random.default_rng(random_seed)
    obj = CountedObjective(spec["func"], dim)
    x_best, f_best, hist, ev, frames = ALGORITHMS[algorithm](
        obj, dim, spec["bounds"], rng, max_evals, record, **algo_params)
    assert obj.raw_calls == obj.n_f + obj.cost_grad * obj.n_grad, "conteo inconsistente"
    return OptimizeResult(problem, algorithm, dim, random_seed,
                          np.asarray(x_best, dtype=float), float(f_best),
                          obj.n_f, obj.n_grad, obj.n_equiv, hist, ev, frames)


# ======================================================================
# 6. Estadísticas sobre varias corridas independientes
# ======================================================================
def run_batch(algorithm, problem, dim, n_runs, seed0, max_evals, threshold, params):
    res = [optimize(algorithm, problem, dim, seed0 + i, max_evals, **params)
           for i in range(n_runs)]
    f = np.array([r.f_best for r in res])
    ok = f < threshold
    to_success = []
    for r in res:
        h = np.asarray(r.history)
        if (h < threshold).any():
            to_success.append(r.eval_trace[int(np.argmax(h < threshold))])
    return {
        "algorithm": algorithm, "dim": dim, "runs": n_runs,
        "mean": f.mean(), "std": f.std(ddof=1) if n_runs > 1 else 0.0,
        "median": float(np.median(f)), "best": f.min(), "worst": f.max(),
        "success_rate": ok.mean(),
        "mean_n_f": np.mean([r.n_f for r in res]),
        "mean_n_grad": np.mean([r.n_grad for r in res]),
        "mean_n_equiv": np.mean([r.n_equiv for r in res]),
        "mean_evals_to_success": float(np.mean(to_success)) if to_success else float("nan"),
    }


# ======================================================================
# 7. GIF (solo dim=2)
# ======================================================================
def make_gif(algorithm, out, dim=2, seed=42, max_evals=20000, params=None,
             max_frames=80, fps=10, problem="rosenbrock"):
    if dim != 2:
        raise ValueError("El GIF solo está implementado para dim=2.")
    if fps < 1 or max_frames < 1:
        raise ValueError('fps y max_frames deben ser positivos')
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    res = optimize(algorithm, problem, dim, seed, max_evals, record=True, **(params or {}))
    if res.frames is None:
        raise ValueError(f"'{algorithm}' no guarda trayectoria; usa otro algoritmo.")
    lo, hi = PROBLEMS[problem]["bounds"]
    g = np.linspace(lo, hi, 300)
    X, Y = np.meshgrid(g, g)
    Z = 100.0 * (Y - X ** 2) ** 2 + (1.0 - X) ** 2

    fig, ax = plt.subplots(figsize=(5.5, 5.4))
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.02, 'Fuente: elaboración propia; funcion1.py, Rosenbrock.\n'
             f'Semilla: {seed}; presupuesto: {max_evals}; algoritmo: {algorithm}\n'
             f'Condición inicial GD: {(params or {}).get("x0", "aleatoria por semilla")}',
             ha='center', fontsize=7)
    ax.contourf(X, Y, Z, levels=np.logspace(-2, 3.5, 25), extend="both", cmap="viridis")
    ax.plot(1, 1, "r*", ms=14, label="óptimo (1,1)")
    is_path = np.ndim(res.frames[0]) == 1
    if is_path:
        line, = ax.plot([], [], "w.-", lw=1, ms=3)
        pt, = ax.plot([], [], "wo", mec="k")
    else:
        sc = ax.scatter([], [], c="white", edgecolors="k", s=25)
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel("$x_1$"), ax.set_ylabel("$x_2$")
    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")

    sel = np.unique(np.linspace(0, len(res.frames) - 1, max_frames).astype(int))

    def update(k):
        i = sel[k]
        if is_path:
            P = np.array(res.frames[: i + 1])
            line.set_data(P[:, 0], P[:, 1])
            pt.set_data([P[-1, 0]], [P[-1, 1]])
        else:
            sc.set_offsets(res.frames[i])
        title.set_text(f"{algorithm} | evals={res.eval_trace[i]} | f_best={res.history[i]:.2e}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=len(sel)).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    metadata = dict(problem=problem, algorithm=algorithm, seed=seed, dim=dim,
                    max_evals=max_evals, params=params or {}, f_best=res.f_best,
                    n_equiv=res.n_equiv, source='Elaboración propia; funcion1.py')
    Path(str(out)+'.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return res


# ======================================================================
# 8. Verificaciones
# ======================================================================
def run_checks() -> bool:
    ok = True
    spec = PROBLEMS["rosenbrock"]
    rng = np.random.default_rng(0)
    for d in (2, 3):
        f_opt = spec["func"](spec["known_optimum"](d))
        print(f"[{'OK' if f_opt == 0 else 'FALLA'}] f(1,...,1) = {f_opt} en {d}D")
        ok &= f_opt == 0
        obj = CountedObjective(spec["func"], d)
        errs = []
        for _ in range(100):
            x = rng.uniform(*spec["bounds"], d)
            errs.append(np.max(np.abs(obj.gradient(x) - spec["grad"](x))))
        e = max(errs)
        print(f"[{'OK' if e < 1e-3 else 'FALLA'}] gradiente numérico vs analítico, {d}D: "
              f"error máx = {e:.2e}")
        ok &= e < 1e-3
    for a in MAIN_ALGOS:
        r = optimize(a, "rosenbrock", 2, 1, 5000)
        c = r.n_equiv <= 5000
        print(f"[{'OK' if c else 'FALLA'}] {a}: equivalentes={r.n_equiv} <= 5000 "
              f"(n_f={r.n_f}, n_grad={r.n_grad})")
        ok &= c
    r1 = optimize("pso", random_seed=7)
    r2 = optimize("pso", random_seed=7)
    same = r1.f_best == r2.f_best
    print(f"[{'OK' if same else 'FALLA'}] misma semilla => mismo resultado")
    ok &= same
    return bool(ok)


# ======================================================================
# 9. CLI
# ======================================================================
def _val(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def parse_overrides(raw, selected):
    """--param k=v aplica a los algoritmos seleccionados que acepten k;
    --param algo.k=v aplica solo a ese algoritmo."""
    out = {a: {} for a in ALGORITHMS}
    for item in raw or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"--param debe tener la forma clave=valor, se recibió '{item}'")
        if "." in key:
            a, k = key.split(".", 1)
            if a not in ALGORITHMS or k not in accepted_params(a):
                raise SystemExit(f"Parámetro desconocido: {key}")
            out[a][k] = _val(val)
        else:
            targets = [a for a in selected if key in accepted_params(a)]
            if not targets:
                raise SystemExit(f"Ningún algoritmo seleccionado acepta '{key}'")
            for a in targets:
                out[a][key] = _val(val)
    return out


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="archivo JSON con valores por defecto")
    common.add_argument("--max-evals", dest="max_evals", type=int)
    common.add_argument('--x0', nargs='+', type=float,
                        help='coordenadas iniciales de GD; ejemplo --x0 -1.2 1')
    common.add_argument("--param", action="append", metavar="[algo.]clave=valor")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verificaciones rápidas de matemática y conteo")

    r = sub.add_parser("run", parents=[common], help="una corrida")
    r.add_argument("--algo", choices=sorted(ALGORITHMS))
    r.add_argument("--dim", type=int)
    r.add_argument("--seed", type=int)

    s = sub.add_parser("stats", parents=[common], help="estadísticas de N corridas")
    s.add_argument("--algos", nargs="+", help="nombres o 'all' (incluye líneas base)")
    s.add_argument("--dims", nargs="+", type=int)
    s.add_argument("--runs", type=int)
    s.add_argument("--seed0", type=int)
    s.add_argument("--threshold", type=float)
    s.add_argument("--out")

    g = sub.add_parser("gif", parents=[common], help="GIF (dim=2)")
    g.add_argument("--algo", choices=sorted(ALGORITHMS))
    g.add_argument("--seed", type=int)
    g.add_argument("--out")
    g.add_argument("--fps", type=int)
    g.add_argument("--max-frames", dest="max_frames", type=int)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "check":
        sys.exit(0 if run_checks() else 1)

    cfg = json.loads(Path(args.config).read_text()) if args.config else {}

    def pick(name, default):
        v = getattr(args, name, None)
        return v if v is not None else cfg.get(name, default)

    max_evals = pick("max_evals", 20000)

    def params_for(selected):
        base = {a: dict(cfg.get("params", {}).get(a, {})) for a in ALGORITHMS}
        for a, d in parse_overrides(args.param, selected).items():
            base[a].update(d)
        punto = getattr(args, 'x0', None)
        if punto is None:
            punto = base['gradient_descent'].get('x0', cfg.get('x0'))
        if punto is not None:
            if 'gradient_descent' not in selected:
                raise SystemExit('--x0 solo se aplica a gradient_descent')
            base['gradient_descent']['x0'] = punto
        return base

    if args.cmd == "run":
        algo, dim, seed = pick("algo", "gradient_descent"), pick("dim", 2), pick("seed", 42)
        print(optimize(algo, "rosenbrock", dim, seed, max_evals, **params_for([algo])[algo]))

    elif args.cmd == "gif":
        algo, seed = pick("algo", "gradient_descent"), pick("seed", 42)
        out = pick("out", f"figs/{algo}_rosenbrock_2d.gif")
        res = make_gif(algo, out, 2, seed, max_evals, params_for([algo])[algo],
                       pick("max_frames", 80), pick("fps", 10))
        print(f"GIF guardado en {out}\n{res}")

    elif args.cmd == "stats":
        algos = pick("algos", list(MAIN_ALGOS))
        if algos == ["all"]:
            algos = sorted(ALGORITHMS)
        dims = pick("dims", [2, 3])
        n_runs, seed0 = pick("runs", 30), pick("seed0", 0)
        thr, out = pick("threshold", 1e-4), pick("out", "results")
        P = params_for(algos)
        rows, t0 = [], time.time()
        print(f"Rosenbrock | corridas={n_runs} | presupuesto={max_evals} evals equivalentes "
              f"| éxito: f < {thr:g} | semillas {seed0}..{seed0 + n_runs - 1}")
        for d in dims:
            for a in algos:
                rows.append(run_batch(a, "rosenbrock", d, n_runs, seed0, max_evals, thr, P[a]))
        hdr = (f"{'algoritmo':24s}{'dim':>4s}{'media':>11s}{'std':>11s}{'mejor':>11s}"
               f"{'peor':>11s}{'éxito':>7s}{'n_f':>9s}{'n_grad':>8s}{'equiv':>9s}")
        print(hdr)
        for r in rows:
            print(f"{r['algorithm']:24s}{r['dim']:>4d}{r['mean']:>11.2e}{r['std']:>11.2e}"
                  f"{r['best']:>11.2e}{r['worst']:>11.2e}{r['success_rate']:>7.0%}"
                  f"{r['mean_n_f']:>9.0f}{r['mean_n_grad']:>8.0f}{r['mean_n_equiv']:>9.0f}")
        Path(out).mkdir(parents=True, exist_ok=True)
        path = Path(out) / "stats_rosenbrock.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {path}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
