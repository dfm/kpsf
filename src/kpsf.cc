#include <climits>
#include <ceres/ceres.h>
#include "kpsf.h"

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

using kpsf::L2Residual;
using kpsf::MixtureBasis;
using kpsf::SumToOneResidual;
using kpsf::MixturePixelResidual;

int kpsf::photometry (const int maxiter, MixtureBasis* basis, double loss_scale,
                      double sum_to_one_strength, double psf_l2_strength,
                      double flat_reg_strength,
                      const int nt, const int npix,
                      const int* x, const int* y,
                      const double* flux_imgs, const double* ferr_imgs,
                      double* coeffs, double* coords, double* ff, double* bg)
{
    // Find the maximum and minimum pixel coordinate values.
    int minx = 0, maxx = 0, miny = 0, maxy = 0;
    for (int i = 0; i < npix; ++i) {
        if (x[i] < minx) minx = x[i];
        if (x[i] > maxx) maxx = x[i];
        if (y[i] < miny) miny = y[i];
        if (y[i] > maxy) maxy = y[i];
    }

    // Build the residual blocks.
    Problem problem;
    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < npix; ++i) {
            int ind = t*npix+i;
            MixturePixelResidual* res =
                new MixturePixelResidual (basis, x[i], y[i],
                                          minx, maxx, miny, maxy,
                                          flux_imgs[ind],
                                          ferr_imgs[ind]);
            CostFunction* cost =
                new AutoDiffCostFunction<MixturePixelResidual, 1, 3,
                                         N_PSF_BASIS, 1, 1> (res);
            // ceres::SoftLOneLoss* loss =
            //     new ceres::SoftLOneLoss(loss_scale);
            problem.AddResidualBlock(cost, NULL,
                                     &(coords[3*t]), coeffs, &(bg[t]),
                                     &(ff[i]));
        }
    }

    // Add regularization terms.
    CostFunction* sum_to_one =
        new AutoDiffCostFunction<SumToOneResidual, 1, N_PSF_BASIS> (
            new SumToOneResidual(N_PSF_BASIS, sum_to_one_strength));
    problem.AddResidualBlock(sum_to_one, NULL, coeffs);

    CostFunction* l2_coeffs =
        new AutoDiffCostFunction<L2Residual, N_PSF_BASIS, N_PSF_BASIS> (
            new L2Residual(N_PSF_BASIS, 0.0, psf_l2_strength));
    problem.AddResidualBlock(l2_coeffs, NULL, coeffs);

    for (int i = 0; i < npix; ++i) {
        CostFunction* l2_flat =
            new AutoDiffCostFunction<L2Residual, 1, 1> (
                new L2Residual(1, 1.0, flat_reg_strength));
        problem.AddResidualBlock(l2_flat, NULL, &(ff[i]));
    }

    // Set up the solver.
    Solver::Options options;
    options.max_num_iterations = maxiter;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.dense_linear_algebra_library_type = ceres::LAPACK;
    options.minimizer_progress_to_stdout = true;

    // Do the solve and report the results.
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    return 0;
}
