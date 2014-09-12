#include <ceres/ceres.h>
#include "psf.h"
#include "residual.h"

using ceres::Solve;
using ceres::Solver;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;

using kpsf::CalibratedPixelResidual;

int photometry_one (const int npix, const double* xpix, const double* ypix,
                    const double* flux, const double* ferr,
                    double* coords, double* coeffs, double* bg)
{
    int i;
    Problem problem;
    for (i = 0; i < npix; ++i) {
        CalibratedPixelResidual* res =
            new CalibratedPixelResidual(xpix[i], ypix[i], flux[i], 1.0 / ferr[i]);
        CostFunction* cost =
            new AutoDiffCostFunction<CalibratedPixelResidual, 1, 3, 3, 1> (res);
        // ceres::SoftLOneLoss* loss =
        //     new ceres::SoftLOneLoss(loss_scale);
        problem.AddResidualBlock(cost, NULL, coords, coeffs, bg);
    }
    return 0;
}

int main () {
    return 0;
}
