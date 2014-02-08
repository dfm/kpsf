#include <vector>
#include <iostream>

#include <ceres/ceres.h>

#include "tpf.h"
#include "psf.h"

using std::vector;

using Eigen::MatrixXd;
using Eigen::VectorXd;

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

using namespace kpsf;

int main ()
{
    int status;

    // Load the Target Pixel file.
    long ccd;
    vector<MatrixXd> flux;
    vector<double> time;
    status = load_tpf("../data/kplr009002278-2010174085026_lpd-targ.fits.gz",
                      &flux, &time, &ccd);
    if (status) return status;

    // Load the PSF basis.
    MixtureBasis* basis =
        new MixtureBasis("../data/kplr07.4_2011265_prf.mog.fits");

    // Allocate the parameter lists.
    int nt = flux.size(),
        nx = flux[0].rows(),
        ny = flux[0].cols();
    MatrixXd flat = MatrixXd::Ones(nx, ny);
    vector<VectorXd> coords(nt);
    double coeffs[] = {0.2, 0.2, 0.2, 0.2, 0.2},
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
                    MixturePixelResidual* res =
                        new MixturePixelResidual (basis, i, j, flux[t](i, j));
                    CostFunction* cost =
                        new AutoDiffCostFunction<MixturePixelResidual, 1, 3,
                                                 N_PSF_BASIS, 1, 1> (res);
                    CauchyLoss* loss = new CauchyLoss(200.0);
                    problem.AddResidualBlock(cost, loss,
                                             &(coords[t](0)), coeffs, &bg,
                                             &(flat(i, j)));
                }
            }
        }
    }

    // Add regularization terms.
    CostFunction* sum_to_one =
        new AutoDiffCostFunction<SumToOneResidual, 1, N_PSF_BASIS> (
            new SumToOneResidual(N_PSF_BASIS, 0.01));
    problem.AddResidualBlock(sum_to_one, NULL, coeffs);

    CostFunction* l2_coeffs =
        new AutoDiffCostFunction<L2Residual, N_PSF_BASIS, N_PSF_BASIS> (
            new L2Residual(N_PSF_BASIS, 0.0, 0.01));
    problem.AddResidualBlock(l2_coeffs, NULL, coeffs);

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (flux[0](i, j) >= 0.0) {
                CostFunction* l2_flat =
                    new AutoDiffCostFunction<L2Residual, 1, 1> (
                        new L2Residual(1, 1.0, 0.01));
                problem.AddResidualBlock(l2_flat, NULL, &(flat(i, j)));
            }
        }
    }

    // Set up the solver.
    Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.dense_linear_algebra_library_type = ceres::LAPACK;
    options.minimizer_progress_to_stdout = true;

    // Do the solve and report the results.
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    // Save the results.
    status = write_results ("../test.fits", time, flat, coords, bg, coeffs);

    delete basis;
    return status;
}
