#include <ceres/ceres.h>
#include "kpsf.h"

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

using kpsf::L2Residual;
using kpsf::MixtureBasis;
using kpsf::SumToOneResidual;
using kpsf::MixturePixelResidual;

int kpsf::photometry (MixtureBasis* basis,
                      const int nt, const int nx, const int ny,
                      const double* flux_imgs, const double* ferr_imgs,
                      double* coeffs, double* coords, double* ff, double* bg)
{
    // Build the residual blocks.
    Problem problem;
    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int ind = (t*nx+i)*ny+j;
                if (flux_imgs[ind] >= 0.0) {
                    MixturePixelResidual* res =
                        new MixturePixelResidual (basis, i, j,
                                                  flux_imgs[ind],
                                                  ferr_imgs[ind]);
                    CostFunction* cost =
                        new AutoDiffCostFunction<MixturePixelResidual, 1, 3,
                                                 N_PSF_BASIS, 1, 1> (res);
                    ceres::SoftLOneLoss* loss = new ceres::SoftLOneLoss(30.0);
                    /* CauchyLoss* loss = new CauchyLoss(90.0); */
                    problem.AddResidualBlock(cost, loss,
                                             &(coords[3*t]), coeffs, bg,
                                             &(ff[i*ny+j]));
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
            int ind = i*ny+j;
            if (flux_imgs[ind] >= 0.0) {
                CostFunction* l2_flat =
                    new AutoDiffCostFunction<L2Residual, 1, 1> (
                        new L2Residual(1, 1.0, 0.01));
                problem.AddResidualBlock(l2_flat, NULL, &(ff[i*ny+j]));
            }
        }
    }

    // Set up the solver.
    Solver::Options options;
    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.dense_linear_algebra_library_type = ceres::LAPACK;
    options.minimizer_progress_to_stdout = true;

    // Do the solve and report the results.
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    return 0;
}
