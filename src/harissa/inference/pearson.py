import numpy as np
from scipy import stats

from harissa.core.inference import Inference, NetworkParameter, Dataset

class Pearson(Inference):
    @property    
    def directed(self) -> bool:
        return False
    
    def run(self,
        dataset: Dataset,
        param: NetworkParameter
    ) -> Inference.Result:
        n_gene_stim = dataset.count_matrix.shape[1]
        score = np.zeros((n_gene_stim, n_gene_stim))

        for i in range(n_gene_stim):
            for j in range(n_gene_stim):
                score[i, j] = stats.pearsonr(
                    dataset.count_matrix[:, i],
                    dataset.count_matrix[:, j]
                ).statistic

        param.interaction[:] = score

        return self.Result(param)