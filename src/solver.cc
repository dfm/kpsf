#include <vector>
#include <iostream>

#include <ceres/ceres.h>

#include "prf.h"
#include "tpf.h"
#include "residual.h"

using std::vector;

using Eigen::MatrixXd;
using Eigen::VectorXd;

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

using namespace kpsf;

int main ()
{
    int status;

    // Load the Target Pixel file.
    long ccd;
    vector<MatrixXd> flux, ferr;
    status = load_tpf("../data/kplr009002278-2010174085026_lpd-targ.fits.gz", &flux, &ferr, &ccd);
    if (status) return status;

    // Load the PRF files.
    vector<MatrixXd> prfs;
    status = load_prfs("../data/kplr07.4_2011265_prf.fits", &prfs);
    if (status) return status;

    // Allocate the parameter lists.
    int nt = flux.size(),
        nx = flux[0].rows(),
        ny = flux[0].cols();
    MatrixXd flat = MatrixXd::Ones(nx, ny);
    vector<VectorXd> coords(nt);
    double coeffs[] = {0.2, 0.2, 0.2, 0.2},
           bg = 0.0;

    // Initialize the coordinates.
    double w;
    for (int t = 0; t < nt; ++t) {
        coords[t] = VectorXd::Zero(3);
        coords[t][0] = flux[t].sum();

        // Compute center of mass.
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                w = flux[t](i, j);
                if (w > 0.0) {
                    coords[t][1] += i * w;
                    coords[t][2] += j * w;
                }
            }
        }

        coords[t][1] /= coords[t][0];
        coords[t][2] /= coords[t][0];
    }

    // Build the residual blocks.
    Problem problem;
    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (flux[t](i, j) >= 0.0) {
                    KeplerPSFResidual* res = new KeplerPSFResidual (i, j, flux[t](i, j), ferr[t](i, j), 3, &prfs);
                    CostFunction* cost =
                        new AutoDiffCostFunction<KeplerPSFResidual, 1, 3, N_PSF_BASIS-1, 1, 1> (res);
                    problem.AddResidualBlock(cost, NULL, &(coords[t](0)), coeffs, &bg, &(flat(i, j)));

                    /* double value; */
                    /* (*res)(&(coords[t](0)), coeffs, &bg, &(flat(i, j)), &value); */
                    /* std::cout << value << " "; */
                }
            }
            /* std::cout << std::endl; */
        }
        /* return 0; */
    }

    // Set up the solver.
    Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.dense_linear_algebra_library_type = ceres::LAPACK;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    return 0;
}
