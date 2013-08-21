#include <iostream>
#include "psf.h"
#include "image.h"

using namespace KPSF;
using Eigen::MatrixXi;
using Eigen::MatrixXd;

int main ()
{
    int w = 10, h = 16;
    PSF psf;
    MatrixXi mask = MatrixXi::Ones(w, h);
    MatrixXd bias = MatrixXd::Zero(w, h),
             ff = MatrixXd::Ones(w, h);

    Image<PSF> img (w, h, &psf, &mask, &bias, &ff);

    std::cout << img.evaluate(0.1, 4.5, 1.0) << std::endl;

    return 0;
}
