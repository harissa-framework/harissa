from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from numpy import *
from operator import itemgetter
from multiprocessing import Pool

from harissa.core import Inference, Dataset, NetworkParameter

def compute_feature_importances(estimator):
    if isinstance(estimator, BaseDecisionTree):
        return estimator.tree_.compute_feature_importances(normalize=False)
    else:
        importances = [
            e.tree_.compute_feature_importances(normalize=False)
            for e in estimator.estimators_
        ]
        importances = array(importances)
        return sum(importances,axis=0) / len(estimator)

def get_link_list(
    vim, 
    gene_names=None, 
    regulators='all', 
    max_count='all', 
    file_name=None
):
    
    """Gets the ranked list of (directed) regulatory links.
    
    Parameters
    ----------
    
    vim: numpy array
        Array as returned by the function genie3(), in which the element (i,j) 
        is the score of the edge directed from the i-th gene to the j-th gene. 
        
    gene_names: list of strings, optional
        List of length p, where p is the number of rows/columns in vim,
        containing the names of the genes. 
        The i-th item of gene_names must
        correspond to the i-th row/column of vim. 
        When the gene names are not provided, the i-th gene is named Gi.
        default: None
        
    regulators: list of strings, optional
        List containing the names of the candidate regulators. 
        When a list of regulators is provided,
        the names of all the genes must be provided (in gene_names), 
        and the returned list contains only
        edges directed from the candidate regulators.
        When regulators is set to 'all', any gene can be a candidate regulator.
        default: 'all'
        
    max_count: 'all' or positive integer, optional
        Writes only the first max_count regulatory links of the ranked list. 
        When max_count is set to 'all', all the regulatory links are written.
        default: 'all'
        
    file_name: string, optional
        Writes the ranked list of regulatory links to the file file_name.
        default: None
        
    Returns
    -------
    
    The list of regulatory links, ordered according to the edge score. 
    Auto-regulations do not appear in the list. 
    Regulatory links with a score equal to zero are randomly permuted. 
    In the ranked list of edges, each line has format:
        
        regulator   target gene     score of edge
    """
    
    # Check input arguments      
    if not isinstance(vim, ndarray):
        raise ValueError('vim must be a square array')
    elif vim.shape[0] != vim.shape[1]:
        raise ValueError('vim must be a square array')
        
    n_genes = vim.shape[0]
        
    if gene_names is not None:
        if not isinstance(gene_names,(list,tuple)):
            raise ValueError(
                'input argument gene_names must be a list of gene names'
            )
        elif len(gene_names) != n_genes:
            raise ValueError(
                'input argument gene_names must be a list of length p, '
                'where p is the number of columns/genes in the expression data'
            )
        
    if regulators != 'all':
        if not isinstance(regulators,(list,tuple)):
            raise ValueError(
                'input argument regulators must be a list of gene names'
            )

        if gene_names is None:
            raise ValueError(
                'the gene names must be specified '
                '(in input argument gene_names)'
            )
        else:
            sIntersection = set(gene_names).intersection(set(regulators))
            if not sIntersection:
                raise ValueError(
                    'The genes must contain at least one candidate regulator'
                )
        
    if max_count != 'all' and not isinstance(max_count,int):
        raise ValueError(
            'input argument max_count must be "all" or a positive integer'
        )
        
    if file_name is not None and not isinstance(file_name,str):
        raise ValueError('input argument file_name must be a string')
    
    

    # Get the indices of the candidate regulators
    if regulators == 'all':
        input_idx = range(n_genes)
    else:
        input_idx = [
            i for i, gene in enumerate(gene_names) if gene in regulators
        ]
    
    # Get the non-ranked list of regulatory links
    vInter = [
        (i,j,score) 
        for (i,j),score in ndenumerate(vim) if i in input_idx and i!=j
        ]
    
    # Rank the list according to the weights of the edges        
    vInter_sort = sorted(vInter,key=itemgetter(2),reverse=True)
    nInter = len(vInter_sort)
    
    # Random permutation of edges with score equal to 0
    flag = 1
    i = 0
    while flag and i < nInter:
        (tf_idx,target_idx,score) = vInter_sort[i]
        if score == 0:
            flag = 0
        else:
            i += 1
            
    if not flag:
        items_perm = vInter_sort[i:]
        items_perm = random.permutation(items_perm)
        vInter_sort[i:] = items_perm
        
    # Write the ranked list of edges
    nToWrite = nInter
    if isinstance(max_count,int) and max_count >= 0 and max_count < nInter:
        nToWrite = max_count
        
    if file_name:
    
        outfile = open(file_name,'w')
    
        if gene_names is not None:
            for i in range(nToWrite):
                (tf_idx,target_idx,score) = vInter_sort[i]
                tf_idx = int(tf_idx)
                target_idx = int(target_idx)
                outfile.write('%s\t%s\t%.6f\n' % (
                    gene_names[tf_idx], 
                    gene_names[target_idx], 
                    score
                ))
        else:
            for i in range(nToWrite):
                (tf_idx,target_idx,score) = vInter_sort[i]
                tf_idx = int(tf_idx)
                target_idx = int(target_idx)
                outfile.write('G%d\tG%d\t%.6f\n' % (
                    tf_idx+1,
                    target_idx+1,
                    score
                ))
            
        
        outfile.close()
        
    else:
        
        if gene_names is not None:
            for i in range(nToWrite):
                (tf_idx,target_idx,score) = vInter_sort[i]
                tf_idx = int(tf_idx)
                target_idx = int(target_idx)
        else:
            for i in range(nToWrite):
                (tf_idx,target_idx,score) = vInter_sort[i]
                tf_idx = int(tf_idx)
                target_idx = int(target_idx)
                
                
                



def genie3(
    expr_data,
    gene_names=None,
    regulators='all',
    tree_method='RF',
    K='sqrt',
    n_trees=1000,
    n_threads=1
):
    
    '''Computation of tree-based scores for all putative regulatory links.
    
    Parameters
    ----------
    
    expr_data: numpy array
        Array containing gene expression values. 
        Each row corresponds to a condition and 
        each column corresponds to a gene.
        
    gene_names: list of strings, optional
        List of length p, where p is the number of columns in expr_data, 
        containing the names of the genes. 
        The i-th item of gene_names must correspond to
        the i-th column of expr_data.
        default: None
        
    regulators: list of strings, optional
        List containing the names of the candidate regulators. 
        When a list of regulators is provided, 
        the names of all the genes must be provided (in gene_names). 
        When regulators is set to 'all', any gene can be a candidate regulator.
        default: 'all'
        
    tree-method: 'RF' or 'ET', optional
        Specifies which tree-based procedure is used: 
        either Random Forest ('RF') or Extra-Trees ('ET')
        default: 'RF'
        
    K: 'sqrt', 'all' or a positive integer, optional
        Specifies the number of selected attributes at each node of one tree: 
        either the square root of the number of candidate regulators ('sqrt'), 
        the total number of candidate regulators ('all'), 
        or any positive integer.
        default: 'sqrt'
         
    n_trees: positive integer, optional
        Specifies the number of trees grown in an ensemble.
        default: 1000
    
    n_threads: positive integer, optional
        Number of threads used for parallel computing
        default: 1
        
        
    Returns
    -------

    An array in which the element (i,j) is the score of the edge directed 
    from the i-th gene to the j-th gene. 
    All diagonal elements are set to zero 
    (auto-regulations are not considered). 
    When a list of candidate regulators is provided, 
    the scores of all the edges directed from a gene 
    that is not a candidate regulator are set to zero.
        
    '''
    
    # Check input arguments
    if not isinstance(expr_data, ndarray):
        raise ValueError(
            'expr_data must be an array in which each row corresponds '
            'to a condition/sample and each column corresponds to a gene'
        )
        
    n_genes = expr_data.shape[1]
    
    if gene_names is not None:
        if not isinstance(gene_names,(list,tuple)):
            raise ValueError(
                'input argument gene_names must be a list of gene names'
            )
        elif len(gene_names) != n_genes:
            raise ValueError(
                'input argument gene_names must be a list of length p, '
                'where p is the number of columns/genes in the expr_data'
            )
        
    if regulators != 'all':
        if not isinstance(regulators,(list,tuple)):
            raise ValueError(
                'input argument regulators must be a list of gene names'
            )

        if gene_names is None:
            raise ValueError(
                'the gene names must be specified '
                '(in input argument gene_names)'
            )
        else:
            sIntersection = set(gene_names).intersection(set(regulators))
            if not sIntersection:
                raise ValueError(
                    'the genes must contain at least one candidate regulator'
                )
        
    if tree_method != 'RF' and tree_method != 'ET':
        raise ValueError(
            'input argument tree_method must be "RF" (Random Forests) '
            'or "ET" (Extra-Trees)'
        )
        
    if K != 'sqrt' and K != 'all' and not isinstance(K,int): 
        raise ValueError(
            'input argument K must be "sqrt", '
            '"all" or a strictly positive integer'
        )
        
    if isinstance(K, int) and K <= 0:
        raise ValueError(
            'input argument K must be "sqrt", '
            '"all" or a strictly positive integer'
        )
    
    if not isinstance(n_trees, int):
        raise ValueError(
            'input argument n_trees must be a strictly positive integer'
        )
    elif n_trees <= 0:
        raise ValueError(
            'input argument n_trees must be a strictly positive integer'
        )
        
    if not isinstance(n_threads,int):
        raise ValueError(
            'input argument n_threads must be a strictly positive integer'
        )
    elif n_threads <= 0:
        raise ValueError(
            'input argument n_threads must be a strictly positive integer'
        )
        
    
    # Get the indices of the candidate regulators
    if regulators == 'all':
        input_idx = list(range(n_genes))
    else:
        input_idx = [
            i for i, gene in enumerate(gene_names) if gene in regulators
        ]

    
    # Learn an ensemble of trees for each target gene,
    # and compute scores for candidate regulators
    vim = zeros((n_genes,n_genes))
    
    if n_threads > 1:

        input_data = list()
        for i in range(n_genes):
            input_data.append( [expr_data,i,input_idx,tree_method,K,n_trees] )

        pool = Pool(n_threads)
        all_output = pool.map(wr_genie3_single, input_data)
    
        for (i,vi) in all_output:
            vim[i,:] = vi

    else:
        for i in range(n_genes):
            
            vi = genie3_single(expr_data,i,input_idx,tree_method,K,n_trees)
            vim[i,:] = vi

   
    vim = transpose(vim)

    return vim
    
    
    
def wr_genie3_single(args):
    return [
        args[1], 
        genie3_single(args[0], args[1], args[2], args[3], args[4], args[5])
    ]
    


def genie3_single(expr_data,output_idx,input_idx,tree_method,K,n_trees):
    
    n_genes = expr_data.shape[1]
    
    # Expression of target gene
    output = expr_data[:,output_idx]
    
    # Normalize output data
    output = output / std(output)
    
    # Remove target gene from candidate regulators
    input_idx = input_idx[:]
    if output_idx in input_idx:
        input_idx.remove(output_idx)

    expr_data_input = expr_data[:,input_idx]
    
    # Parameter K of the tree-based method
    if (K == 'all') or (isinstance(K,int) and K >= len(input_idx)):
        max_features = "auto"
    else:
        max_features = K
    
    if tree_method == 'RF':
        treeEstimator = RandomForestRegressor(
            n_estimators=n_trees,
            max_features=max_features
        )
    elif tree_method == 'ET':
        treeEstimator = ExtraTreesRegressor(
            n_estimators=n_trees,
            max_features=max_features
        )

    # Learn ensemble of trees
    treeEstimator.fit(expr_data_input,output)
    
    # Compute importance scores
    feature_importances = compute_feature_importances(treeEstimator)
    vi = zeros(n_genes)
    vi[input_idx] = feature_importances
       
    return vi


class Genie3(Inference):
    def run(self,
        data: Dataset,
        param: NetworkParameter
    ) -> Inference.Result:
        param.interaction[:] = genie3(data.count_matrix)
        return Inference.Result(param)
    
    @property    
    def directed(self) -> bool:
        return True