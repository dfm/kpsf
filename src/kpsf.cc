#include <iostream>
#include "ceres/ceres.h"
#include "psf.h"
#include "image.h"

#define WIDTH 10
#define HEIGHT 16

using namespace KPSF;
using Eigen::MatrixXi;
using Eigen::MatrixXd;

struct ExponentialResidual {
    ExponentialResidual (MatrixXd flux, Image<PSF> img)
        : flux_ (flux), img_ (img) {}

    template <typename T>
    bool operator() (const T* const x, const T* const y, const T* const f,
                     T* residual) const {
        int ix, iy, width = flux_.rows(), height = flux_.cols();
        std::cout << width << " " << height << std::endl;
        for (ix = 0; ix < width; ++ix)
            for (iy = 0; iy < height; ++iy)
                residual[ix * height + iy] = T(flux_(ix, iy)) - img_.evaluate(x[0], y[0], f[0], ix, iy);

        return true;
    }

private:
    const MatrixXd flux_;
    const Image<PSF> img_;

};

int main ()
{
    PSF psf;
    MatrixXi mask = MatrixXi::Ones(WIDTH, HEIGHT);
    MatrixXd bias = MatrixXd::Zero(WIDTH, HEIGHT),
             ff = MatrixXd::Ones(WIDTH, HEIGHT);
    Image<PSF> img (WIDTH, HEIGHT, &psf, &mask, &bias, &ff);
    double x0 = 2.5, y0 = 4.5, f0 = 1.0;
    MatrixXd data = img.generate(x0, y0, f0);

    x0 += 0.1;
    y0 -= 0.1;
    f0 -= 0.2;

    ceres::Problem problem;
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<ExponentialResidual, WIDTH*HEIGHT, 1, 1, 1>(
                new ExponentialResidual(data, img));
    problem.AddResidualBlock(cost_function, NULL, &x0, &y0, &f0);

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << "Final   x: " << x0 << " y: " << y0 << " f: " << f0 << "\n";

    return 0;
}
