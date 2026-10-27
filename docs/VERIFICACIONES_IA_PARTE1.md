# Registro de verificaciones y errores: parte 1

Esta sección aporta evidencia real de las correcciones. No se atribuyen errores a IA sin conservar la conversación original. No se inventan prompts ni se afirma disponer de cinco hallazgos históricos. El equipo debe añadir prompt literal, respuesta original y su procedencia cuando los tenga.

## 1. Condición inicial no configurable

**Evidencia anterior:** en `4f6beb4:funcion1.py`, `gradient_descent` siempre inicializaba `x = rng.uniform(lo, hi, dim)` y no recibía x0. La interfaz de consola tampoco tenía `--x0`. Coincide con el fallo de ejecución descrito por el profesor.

**Corrección:** `--x0` y parámetro de Python/configuración; controles de dimensión, finitud y dominio. No se reemplaza ni recorta silenciosamente el punto solicitado.

**Verificación:** los tests comparan exactamente la primera posición de la trayectoria contra el punto solicitado en 2D/3D, su valor de función y el presupuesto utilizado.

**Lección:** comprobar la petición de sustentación con condiciones explícitas antes de presentar el programa. Prompt y respuesta original: no disponibles aquí; no se reconstruyen como prueba de autoría.

## 2. Bloque de ejemplo ejecutable y módulo inexistente

**Evidencia anterior:** el archivo terminaba con `from optimizacion import optimize` dentro del bloque principal, seguido de una corrida de ejemplo al nivel global. No existe `optimizacion.py` en esta versión del repositorio. Una ejecución normal podía completar el cálculo y luego fallar en esa importación; importar el módulo también ejecutaba la corrida global de ejemplo.

**Corrección:** se retiró el ejemplo ejecutable suelto y se conservó `if __name__ == '__main__': main()` como único punto de entrada. Los ejemplos de uso quedaron documentados fuera del código ejecutable.

**Verificación:** importación de `funcion1` sin ejecutar una corrida; comando run termina con código 0; `check` termina con sus comprobaciones correctas.

**Lección:** probar tanto la interfaz de consola como la importación de funciones reutilizables. Procedencia de IA: no establecida; falta conversación original.

## 3. Comandos documentados con un nombre de archivo incorrecto

**Evidencia anterior:** el docstring indicaba `python optimizacion.py ...`, pero el archivo real es `funcion1.py`.

**Corrección:** ejemplos actualizados y ejecutados con el nombre real. Es una inconsistencia de documentación, no un fallo matemático.

**Lección:** ejecutar literalmente los comandos que aparecen en la documentación. No se cuenta este registro como cinco hallazgos ni se inventan categorías para completar una cuota.

## Verificaciones matemáticas satisfactorias

Se mantiene el chequeo de `f(1,...,1)=0` en 2D y 3D; se compara gradiente central contra analítico y se verifica que no se dupliquen las llamadas internas del gradiente en el presupuesto equivalente. Estas comprobaciones son verificaciones positivas, no alucinaciones descubiertas.

La redacción final debe distinguir errores de código, errores documentales y verificaciones sin error. La publicación de este registro en el blog y la incorporación de las conversaciones originales son pasos editoriales del equipo.
