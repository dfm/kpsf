#include <iostream>
#include "ceres/ceres.h"

#include "solver.h"
#include "pixel_residual.h"

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

int kpsf_solve (int ntime, int *npixels, double ***data, double *dim,
                double **coords, double *psfpars)
{
    int i, j;
    Problem problem;

    for (i = 0; i < ntime; ++i) {
        for (j = 0; j < npixels[i]; ++j) {
            CostFunction *cost =
                new AutoDiffCostFunction<PixelResidual, 1, 3, 3> (
                    new PixelResidual(dim, data[i][j]));
            problem.AddResidualBlock (cost, NULL, coords[i], psfpars);
        }
    }

    Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    return 0;
}
