import pathlib

import ecole


SOURCE_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = SOURCE_DIR / "vendor/ecole/libecole/tests/data"


def get_model():
    """Return a Model object with a valid problem."""
    model = ecole.scip.Model.from_file(str(DATA_DIR / "bppc8-02.mps"))
    model.disable_cuts()
    model.disable_presolve()
    model.set_param("randomization/permuteconss", True)
    model.set_param("randomization/permutevars", True)
    model.set_param("randomization/permutationseed", 784)
    model.set_param("randomization/randomseedshift", 784)
    model.set_param("randomization/lpseed", 784)
    return model
