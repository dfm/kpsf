#include <iostream>
#include <ceres/ceres.h>

#include "kpsf.h"
#include "psf.h"
#include "residual.h"

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

using kpsf::PixelResidual;
using kpsf::CalibratedPixelResidual;
using kpsf::GaussianPrior;

// int kpsf::photometry_one (const int npix, const double* xpix,
//                           const double* ypix, const double* flux,
//                           const double* ferr, double* coords, double* coeffs,
//                           double* bg)
// {
//     int i;
//     Problem problem;
//     for (i = 0; i < npix; ++i) {
//         CalibratedPixelResidual* res =
//             new CalibratedPixelResidual(xpix[i], ypix[i], flux[i], 1.0 / ferr[i]);
//         CostFunction* cost =
//             new AutoDiffCostFunction<CalibratedPixelResidual, 1, 3, 9, 1> (res);
//         // ceres::SoftLOneLoss* loss =
//         //     new ceres::SoftLOneLoss(loss_scale);
//         problem.AddResidualBlock(cost, NULL, coords, coeffs, bg);
//     }
//
//     // Set up the solver.
//     Solver::Options options;
//     options.max_num_iterations = 100;
//     options.linear_solver_type = ceres::DENSE_QR;
//     // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//     options.dense_linear_algebra_library_type = ceres::LAPACK;
//     options.minimizer_progress_to_stdout = true;
//
//     // Do the solve and report the results.
//     Solver::Summary summary;
//     Solve(options, &problem, &summary);
//     std::cout << summary.BriefReport() << std::endl;
//
//     return 0;
// }

int kpsf::photometry_all (const int nt, const int npix, const double* xpix,
                          const double* ypix, const double* flux,
                          const double* ferr, double* coords, double* coeffs,
                          double* ff, double* bg)
{
    int i, j, ind;
    Problem problem;
    for (i = 0; i < nt; ++i) {
        for (j = 0; j < npix; ++j) {
            ind = i*npix+j;
            PixelResidual* res =
                new PixelResidual(xpix[j], ypix[j], flux[ind], 1./ferr[ind]);
            CostFunction* cost =
                new AutoDiffCostFunction<PixelResidual, 1, 3, 9, 1, 1> (res);
            // ceres::SoftLOneLoss* loss =
            //     new ceres::SoftLOneLoss(loss_scale);
            problem.AddResidualBlock(cost, NULL,
                                     &(coords[i*3]), coeffs, &(bg[i]), &(ff[j]));

            cost = new AutoDiffCostFunction<GaussianPrior, 1, 1> (
                new GaussianPrior(1.0, 0.1));
            problem.AddResidualBlock(cost, NULL, &(ff[j]));
        }
    }

    // CostFunction* cost = new AutoDiffCostFunction<PSFPrior, 1, 2> (new PSFPrior(2.0));
    // problem.AddResidualBlock(cost, NULL, &(coeffs[4]));

    // Set up the solver.
    Solver::Options options;
    options.max_num_iterations = 500;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.dense_linear_algebra_library_type = ceres::LAPACK;
    options.minimizer_progress_to_stdout = true;

    // Do the solve and report the results.
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    return 0;
}
