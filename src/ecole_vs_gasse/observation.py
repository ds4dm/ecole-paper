import learn2branch.utilities as l2b


class NodeBipartite:
    def before_reset(self, model):
        self.buffer = {}

    def extract(self, model, done):
        if done:
            return None
        return l2b.extract_state(model.as_pyscipopt(), self.buffer)


class Khalil2016:
    def before_reset(self, model):
        self.buffer = {}

    def extract(self, model, done):
        if done:
            return None

        pyscipopt_model = model.as_pyscipopt()

        # Initialize root buffer for Khalil features extraction
        if pyscipopt_model.getNNodes() == 1:
            l2b.extract_khalil_variable_features(pyscipopt_model, [], self.buffer)

        cands, *_ = pyscipopt_model.getPseudoBranchCands()
        return l2b.extract_khalil_variable_features(pyscipopt_model, cands, self.buffer)
