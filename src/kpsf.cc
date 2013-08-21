#include <iostream>
#include <cmath>
#include "ceres/ceres.h"

#define WIDTH 10
#define HEIGHT 16

using Eigen::MatrixXi;
using Eigen::MatrixXd;

template <typename T>
class PSF {
public:
    PSF (const T* const pars)
        : pars_ (pars) {
        invdet_ = T(0.5) / (pars[0] * pars[2] - pars[1] * pars[1]);
        factor_ = sqrt(invdet_) / T(M_PI);
    };

    T evaluate (const T dx, const T dy) const {
        return factor_ * exp(-invdet_ * (pars_[2] * dx * dx +
                                         pars_[0] * dy * dy -
                                         T(2) * pars_[1] * dx * dy));
    };

private:
    const T* const pars_;
    T invdet_, factor_;
};

class ImageResidualBlock {
public:
    ImageResidualBlock (MatrixXd flux)
        : flux_ (flux) {};

    template <typename T>
    bool operator() (const T* const coords, const T* const psfpars,
                     T* residual) const {
        int ix, iy,
            width = flux_.rows(), height = flux_.cols();
        PSF<T> psf (psfpars);

        for (ix = 0; ix < width; ++ix)
            for (iy = 0; iy < height; ++iy) {
                T value = coords[2] * psf.evaluate(coords[0] - T(ix),
                                                   coords[1] - T(iy));
                residual[ix * height + iy] = T(flux_(ix, iy)) - value;
            }

        return true;
    }

private:
    const MatrixXd flux_;
};

int main ()
{
    double coords[3] = {3.1234, 4.49815, 1.0},
           psfpars[3] = {1.0, 0.1, 1.0};

    PSF<double> psf(psfpars);
    MatrixXd data(WIDTH, HEIGHT);

    int ix, iy;
    for (ix = 0; ix < WIDTH; ++ix)
        for (iy = 0; iy < HEIGHT; ++iy)
            data(ix, iy) = coords[2] * psf.evaluate(coords[0] - ix,
                                                    coords[1] - iy);

    coords[0] += 1;
    coords[1] -= 1;
    coords[2] += 10;

    psfpars[0] -= 0.5;
    psfpars[1] += 0.5;
    psfpars[2] += 0.5;

    ceres::Problem problem;
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<ImageResidualBlock, WIDTH*HEIGHT, 3, 3>(
                new ImageResidualBlock(data));
    problem.AddResidualBlock(cost_function, NULL, coords, psfpars);

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << "Final   x: " << coords[0] << " y: " << coords[1] << " f: " << coords[2] << "\n";
    std::cout << "PSF: " << psfpars[0] << " " << psfpars[1] << " " << psfpars[2] << "\n";

    return 0;
}
