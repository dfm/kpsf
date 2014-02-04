#ifndef _MODEL_H_
#define _MODEL_H_

#include <cmath>
#include <cfloat>
#include <vector>
#include <Eigen/Dense>
#include <ceres/numeric_diff_functor.h>

#include "prf.hpp"

using std::vector;
using Eigen::MatrixXd;
using ceres::NumericDiffFunctor;

namespace kpsf {

double lanczos_kernel (unsigned int a, double x) {
    double fx, px;
    if (x <= DBL_EPSILON) return 1.0;
    fx = fabs(x);
    if (fx >= a) return 0.0;
    px = M_PI * x;
    return a * sin(px / a) * sin(px) / (px * px);
}

class LanczosInterpolator {

public:

    LanczosInterpolator (int order, vector<MatrixXd> * psf_basis)
        : psf_basis_(psf_basis), order_(order) {};

    bool operator() (const double* coeffs, const double* xi, const double* yi,
                     double* result) const {
        double lx, ly;
        int indx, indy, ixi = round(*xi), iyi = round(*yi);

        *result = 0.0;
        for (indx = -order_; indx <= order_; ++indx) {
            lx = lanczos_kernel (order_, indx - *xi + ixi);
            for (indy = -order_; indy <= order_; ++indy) {
                ly = lanczos_kernel (order_, indy - *yi + iyi);
                *result += lx * ly * evaluate (coeffs, ixi+indx, iyi+indy);
            }
        }

        return true;
    };

private:

    int order_;
    vector<MatrixXd> * psf_basis_;

    double evaluate (const double* coeffs, int i, int j) const {
        int ind;
        double value = 0.0, norm = 1.0;
        for (ind = 0; ind < N_PSF_BASIS - 1; ++ind) {
            value += coeffs[ind] * (*psf_basis_)[ind](i, j);
            norm -= coeffs[ind];
        }
        value += norm * (*psf_basis_)[N_PSF_BASIS-1](i, j);
        return value;
    };

};

class KeplerPSF {

public:

    typedef NumericDiffFunctor<LanczosInterpolator, ceres::CENTRAL, N_PSF_COEFF, 1, 1, 1>
            LanczosInterpolatorFunctor;

    KeplerPSF (int i, int j, int order, vector<MatrixXd> * psf_basis)
        : i_(i), j_(j)
    {
        interpolator_.reset(new LanczosInterpolatorFunctor(new LanczosInterpolator(order, psf_basis)));
    };

    template <typename T>
    T operator() (const T* sx, const T* sy, const T* coeffs, T* result) {
        T xi = T(OVERSAMPLE) * (T(i_) - *sx) + T(CENTER_X),
          yi = T(OVERSAMPLE) * (T(j_) - *sy) + T(CENTER_Y);
        return (*interpolator_)(coeffs, &xi, &yi, result);
    };

private:

    int i_, j_;
    ceres::internal::scoped_ptr<LanczosInterpolatorFunctor> interpolator_;

};

};

#endif
