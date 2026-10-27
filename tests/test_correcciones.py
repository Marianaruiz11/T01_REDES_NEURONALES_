import unittest
import subprocess
import sys
import tempfile
import json
from pathlib import Path
import numpy as np
from funcion1 import optimize, rosenbrock


class CondicionInicialTests(unittest.TestCase):
    def test_respeta_punto_y_presupuesto(self):
        for punto in ([-1.2,1.],[-1.2,1.,.5]):
            r=optimize(dim=len(punto),x0=punto,max_evals=100,record=True)
            np.testing.assert_array_equal(r.frames[0],punto)
            self.assertEqual(r.history[0],rosenbrock(punto))
            self.assertLessEqual(r.n_equiv,100)
            self.assertLessEqual(r.f_best,r.history[0])

    def test_gd_fijo_no_depende_de_semilla(self):
        a=optimize(x0=[-1.2,1],random_seed=1,max_evals=100)
        b=optimize(x0=[-1.2,1],random_seed=99,max_evals=100)
        np.testing.assert_array_equal(a.x_best,b.x_best)
        self.assertEqual(a.n_equiv,b.n_equiv)

    def test_cli_config_y_prioridad(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta=Path(carpeta)/'config.json'
            ruta.write_text(json.dumps(dict(algo='gradient_descent',dim=2,
                                            x0=[0,0],max_evals=1)))
            r=subprocess.run([sys.executable,'funcion1.py','run','--config',str(ruta),
                              '--x0','1','1'],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('f_best  = 0.000000e+00',r.stdout)
            self.assertEqual(r.stdout.count('x_best'),1)

    def test_errores_explicitos(self):
        for punto in ([1],[9,0],[float('nan'),0]):
            with self.assertRaises(ValueError):optimize(x0=punto)
        with self.assertRaises(ValueError):optimize(max_evals=0)


if __name__=='__main__':unittest.main()
