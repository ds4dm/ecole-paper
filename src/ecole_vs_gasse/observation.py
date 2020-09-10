import pyscipopt_gasse as pyscipopt


class NodeBipartite:

    def reset(self, model):
        raise NotImplementedError

    def obtain_observation(self, model):
        pyscipopt_model = model.as_pyscipopt()
        raise NotImplementedError


class Khalil2016:

    def reset(self, model):
        raise NotImplementedError

    def obtain_observation(self, model):
        pyscipopt_model = model.as_pyscipopt()
        raise NotImplementedError

