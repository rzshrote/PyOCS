import cvxpy
import numpy

def solve_ocs(bv, geno, inbmax):
    """
    Solve Optimal Contribution Selection:
                argmax(x) = (bv)'(x)
        subject to:
                    x'Kx <= inbmax
                     x_i >= 0       for all i
        Where:
            'bv' are breeding values.
            'K' is a kinship matrix.
            'inbmax' is the inbreeding constraint.

    Reference
    ---------
        Allier A, Lehermeier C, Charcosset A, Moreau L, Teyssèdre S (2019).
        Improving Short- and Long-Term Genetic Gain by Accounting for
        Within-Family Variance in Optimal Cross-Selection. Front Genet. 10: 1006

    Parameters
    ----------
    bv : numpy.ndarray
        A matrix of shape (n,) containing breeding values.
        Where:
            'n' is the number of breeding individuals.
    geno : numpy.ndarray
        A matrix of shape (n,p) containing genotypes encoded using {-1, 0, 1}.
        Where:
            'n' is the number of breeding individuals.
            'p' is the number of genotypic markers.
    inbmax : float
        Maximum mean population inbreeding constraint. Must be greater than the
        mean kinship for individuals in 'geno'.

    Returns
    -------
    x : numpy.ndarray
        A matrix of shape (n,) containing optimal breeding individual
        contribution proportions for each of the 'n' breeding individuals.
    """
    # calculate kinship matrix K = 1/2 * (1 + (1/n)XX')
    K = 0.5 * (1.0+((1/len(geno)) * geno @ geno.T)) # (n,n)

    # Attempt Cholesky decompose K into K = C'C
    try:
        C = numpy.linalg.cholesky(K).T              # (n,n)
    except numpy.linalg.LinAlgError:
        raise RuntimeError("Kinship matrix is not positive definite")

    # take sqrt of inbreeding constraint
    inbconst = numpy.sqrt(inbmax)

    # define cvxpy variable
    x = cvxpy.Variable(len(geno))                   # (n,)

    # define the objective function
    soc_objfn = cvxpy.Maximize(bv.T @ x)            # max (bv)'(sel)

    # define constraints
    soc_constraints = [
        cvxpy.SOC(inbconst, C @ x),                 # ||C @ x||_2 <= inbconst
        cvxpy.sum(x) == 1.0,                        # sum(x_i) == 1
        x >= 0.0                                    # x_i >= 0 for all i
    ]

    # solve the problem
    sol = prob.solve()

    # check solution quality
    if prob.status != "optimal":
        raise RuntimeError("OCS optimization could not be solved")

    # return optimal allocation
    return x.value
