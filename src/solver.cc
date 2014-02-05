#include <vector>
#include <iostream>

#include <ceres/ceres.h>

#include "prf.h"
#include "tpf.h"
#include "model.h"

using namespace kpsf;

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

int main ()
{
    // Load the PRF files.
    vector<MatrixXd> prfs;
    int status = load_prfs("../data/kplr02.1_2011265_prf.fits", &prfs);
    if (status) return status;

    // Load the Target Pixel file.
    vector<MatrixXd> flux, ferr;
    status = load_tpf("../data/kplr009002278-2010174085026_lpd-targ.fits.gz", &flux, &ferr);
    if (status) return status;

    std::cout << flux[0] << std::endl;

    double coords[] = {1.0, 3.394, 2.195}, coeffs[] = {0.6, 0.1, 0.1, 0.1}, value,
           ff = 1.0, bg = 0.0;

    Problem problem;

    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 5; ++j) {
            CostFunction *cost =
                new AutoDiffCostFunction<KeplerPSFResidual, 1, 3, N_PSF_COEFF, 1, 1> (
                    new KeplerPSFResidual (i, j, 1.0, 0.1, 3, &prfs));
            problem.AddResidualBlock(cost, NULL, coords, coeffs, &bg, &ff);
        }
    }

    return 0;
}
