# Correcciones del trabajo 1: exclusivamente parte 1

Se conserva el protocolo de comparación y el historial real. Los resultados y archivos del punto 2 no se modifican. Las fechas de los nuevos commits corresponden al trabajo de corrección; no se reconstruye ni simula un historial pasado.

## Ejecutar en vivo con la condición inicial solicitada

Desde la raíz del repositorio, con `requirements.txt` instalado:

```bash
python funcion1.py run --algo gradient_descent --dim 2 --x0 -1.2 1 --max-evals 20000
python funcion1.py run --algo gradient_descent --dim 3 --x0 -1.2 1 0.5 --max-evals 20000
python funcion1.py run --config docs/parte1_sustentacion.json
```

`--x0` define exactamente las coordenadas iniciales. Se rechazan dimensiones incorrectas, coordenadas no finitas y puntos fuera del dominio [-2.048,2.048]; no se altera silenciosamente la condición solicitada. Si se omite, se conserva la inicialización aleatoria por semilla. Con punto fijo, GD es determinista en esta implementación: cambiar solo la semilla no crea corridas independientes.

Los parámetros del método se cambian sin editar Python:

```bash
python funcion1.py run --algo gradient_descent --dim 2 --x0 -1.2 1 --param lr0=0.5 --param h=0.00001 --param tol=0.000001 --param shrink=0.8 --param c=0.0001 --param max_ls=40
python funcion1.py stats --algos gradient_descent genetic pso differential_evolution --dims 2 3 --runs 30 --seed0 0 --max-evals 20000 --out results/parte1_corregida
```

Para conservar el protocolo original de múltiples inicializaciones, **no** añada un x0 fijo al comando estadístico. La condición fija se usa para responder a una petición concreta o estudiar sensibilidad. `--param` también permite modificar parámetros de GA, PSO y DE; consulte sus firmas en `funcion1.py`. No se afirma que un presupuesto determinado garantice convergencia desde cualquier condición.

## Verificaciones

```bash
python -m unittest discover -s tests -v
python funcion1.py check
```

Se comprueba que la primera posición registrada sea exactamente x0 en 2D/3D, que su primera evaluación corresponda a Rosenbrock, que GD respete el presupuesto, que no dependa de la semilla con x0 fijo y que rechace entradas inválidas. La verificación existente contrasta gradiente numérico/analítico y el conteo de evaluaciones.

## Animación y fuente

```bash
python funcion1.py gif --algo gradient_descent --x0 -1.2 1 --max-evals 20000 --out results/parte1_corregida/gd_x0.gif
```

El GIF incluye su fuente en la figura y guarda metadatos en `gd_x0.gif.json`. Puede regenerarse para otras condiciones. La fuente es elaboración propia con la función de Rosenbrock, no una imagen tomada de una fuente externa.

Para el blog use un archivo GIF alojado y accesible, no una captura estática ni únicamente un enlace al código. Ejemplo HTML, **ajustando la URL al archivo real publicado**:

```html
<figure>
  <img src="URL_PUBLICA_REAL/gd_x0.gif"
       alt="Figura 1. Trayectoria del descenso por gradiente en Rosenbrock desde (-1.2, 1).">
  <figcaption>Figura 1. Evolución del descenso por gradiente.
  Fuente: elaboración propia con funcion1.py; condición inicial (-1.2, 1),
  semilla 42 y presupuesto 20 000 evaluaciones equivalentes.</figcaption>
</figure>
```

La inclusión en el blog requiere acceso a su editor y comprobar la reproducción en la página publicada; no queda validada solo por generar el GIF localmente. En `PARTE1_ECUACIONES.tex` están las ecuaciones para incorporar al reporte. Cite cada figura en el texto y revise la fuente de figuras históricas antes de reutilizarlas.

## Participación en sustentación

Cada integrante debe practicar cambiar x0, explicar diferencias centrales/Armijo y justificar la comparación por evaluaciones equivalentes. Todos deben poder ejecutar el mismo comando y explicar su salida. La participación oral no se resuelve con una modificación al código.

Se incluye una animación regenerada con x0=(-1.2,1) y presupuesto 20.000 en `docs/figuras/gd_condicion_inicial.gif`, junto con sus metadatos. `docs/parte1_animacion.html` permite verla con título y fuente al abrir el archivo en un navegador. Esta página no se ha publicado en el blog.
