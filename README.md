# Trabajo 1 — Parte 1 y demo TSP

## Notebook del punto 1

Abra [punto_1/Experimentos.ipynb](punto_1/Experimentos.ipynb). Se incorporó la versión del notebook del equipo y se corrigió su módulo acompañante, conservando Rosenbrock, Rastrigin y las animaciones Plotly. Instale sus dependencias con `python -m pip install -r punto_1/requirements.txt` y seleccione ese entorno como kernel. Cambie `X0_2D` y `X0_3D` en la primera celda para la demostración. La corrección no incorpora archivos del punto 2 del repositorio original.

## Corrección de la parte 1

La parte 1 permite ahora elegir la condición inicial del descenso por gradiente. Consulte [instrucciones y verificaciones](docs/CORRECCIONES_PARTE1.md), [ecuaciones para el reporte](docs/PARTE1_ECUACIONES.tex) y [registro de verificaciones](docs/VERIFICACIONES_IA_PARTE1.md).

```bash
python funcion1.py run --algo gradient_descent --dim 2 --x0 -1.2 1 --max-evals 20000
```

La documentación de la demo del punto 2 se conserva a continuación; sus datos y código no se modificaron como parte de esta corrección.

---


# TSP España — paquete DEMO ejecutable

## Qué es esta versión
Esta carpeta contiene una versión, para que el equipo pruebe el software antes de sustituir los datos DEMO por los datos documentados del trabajo.

**Importante:** `rutas_2020.csv` y `rutas_2026.csv` son generadas por `src/generar_demo.py` y son **datos sintéticos de demostración**. No representan tarifas, tiempos ni distancias oficiales de España.

El archivo `data/capitales.xlsx` sí es la base real entregada por el equipo, con las 47 capitales y sus coordenadas.

## 1. Instalar

```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

## 2. Ejecutar la demo completa
Windows:
```bash
run_demo.bat
```

macOS/Linux:
```bash
./run_demo.sh
```

O manualmente:
```bash
python src/generar_demo.py
python src/tsp_espana.py --scenario both --sensitivity
```

## 3. Qué debe aparecer
Se generan:

- `results/resumen_principal.csv`
- `results/contraste_2020_2026.csv`
- `results/sensibilidad_2020.csv`
- `results/sensibilidad_2026.csv`
- gráficos de convergencia ACO/GA
- mapas esquemáticos de las rutas
- GIF de evolución de ACO y GA
- matrices de costos y rutas seleccionadas para cada escenario

## 4. Qué significa la arquitectura
Para cada par de capitales puede haber varias rutas candidatas. El programa calcula:

`costo = valor_hora * tiempo + peaje + combustible`

con:

`combustible = distancia_km * consumo_L_100km/100 * precio_eur_L`

y selecciona la ruta candidata de menor costo generalizado antes de resolver el TSP.

## 5. Para el trabajo real
Después de aprobar esta demo, se reemplazarán los archivos DEMO por:

- `rutas_2020.csv`: datos históricos realmente verificables.
- `rutas_2026.csv`: datos actuales realmente verificables.
- `parametros.json`: vehículo, combustible y valor de la hora con sus fuentes.
