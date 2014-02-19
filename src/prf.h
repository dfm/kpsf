#ifndef _PRF_H_
#define _PRF_H_

#include <cmath>
#include <vector>
#include <cstdio>
#include <fitsio.h>

// Ignore string warnings. It's cfitsio's fault!
#pragma GCC diagnostic ignored "-Wwrite-strings"

#define N_PSF_BASIS 5

using std::vector;

namespace kpsf {

class FixedGaussian {

public:

    FixedGaussian (double amp, double xpos, double ypos, double xvar,
                   double covar, double yvar)
        : amp_(amp), xpos_(xpos), ypos_(ypos), xvar_(xvar), covar_(covar),
          yvar_(yvar)
    {
        det_ = xvar * yvar - covar * covar;
        factor_ = amp_ / (2*M_PI*sqrt(det_));
    };

    double get_det () const { return det_; };

    template <typename T>
    T evaluate (T i, T j) const {
        T dx = T(xpos_) - i, dy = T(ypos_) - j,
          x = T(yvar_) * dx - T(covar_) * dy,
          y = T(xvar_) * dy - T(covar_) * dx,
          v = (dx * x + dy * y) / T(det_);
        return T(factor_) * exp(T(-0.5) * v);
    }

private:

    double amp_, xpos_, ypos_, xvar_, covar_, yvar_, det_, factor_;

};

class Mixture {

public:

    int add_gaussian (double amp, double xpos, double ypos, double xvar,
                      double covar, double yvar)
    {
        FixedGaussian g(amp, xpos, ypos, xvar, covar, yvar);
        if (g.get_det() <= 0.0) return 1;
        gaussians_.push_back(g);
        return 0;
    }

    template <typename T>
    T evaluate (T i, T j) const {
        T value = T(0.0);
        for (int k = 0; k < gaussians_.size(); ++k)
            value += gaussians_[k].evaluate<T> (i, j);
        return value;
    }

private:

    vector<FixedGaussian> gaussians_;

};

class MixturePSF {

public:

    MixturePSF (const char* fn) : basis_(N_PSF_BASIS) {
        fitsfile *f;
        status_ = 0;

        // Open the FITS file.
        if (fits_open_file(&f, fn, READONLY, &status_)) {
            fits_report_error(stderr, status_);
            return;
        }

        // Loop over PSF bases.
        int anynull, hdutype;
        double nullval = -1;
        for (int i = 0; i < N_PSF_BASIS; ++i) {
            // Move to the HDU.
            if (fits_movabs_hdu(f, i+2, &hdutype, &status_)) {
                fits_report_error(stderr, status_);
                return;
            }

            // Get the number of rows in the table.
            long nrows;
            char comm[100];
            if (fits_read_key_lng(f, "NAXIS2", &nrows, comm, &status_)) {
                fits_report_error(stderr, status_);
                return;
            }

            // Initialize the mixture.
            basis_[i] = Mixture();

            // Load the columns.
            double amp, xpos, ypos, xvar, covar, yvar;
            for (int j = 0; j < nrows; ++j) {
                fits_read_col(f, TDOUBLE, 1, j+1, 1, 1, &nullval, &amp, &anynull, &status_);
                fits_read_col(f, TDOUBLE, 2, j+1, 1, 1, &nullval, &xpos, &anynull, &status_);
                fits_read_col(f, TDOUBLE, 3, j+1, 1, 1, &nullval, &ypos, &anynull, &status_);
                fits_read_col(f, TDOUBLE, 4, j+1, 1, 1, &nullval, &xvar, &anynull, &status_);
                fits_read_col(f, TDOUBLE, 5, j+1, 1, 1, &nullval, &covar, &anynull, &status_);
                fits_read_col(f, TDOUBLE, 6, j+1, 1, 1, &nullval, &yvar, &anynull, &status_);

                if (status_) {
                    fits_report_error(stderr, status_);
                    return;
                }

                status_ = basis_[i].add_gaussian(amp, xpos, ypos, xvar, covar, yvar);
                if (status_) {
                    fprintf(stderr, "Invalid Gaussian basis.");
                    return;
                }
            }
        }
    };

    int get_status () const { return status_; };

    template <typename T>
    T evaluate (const T* coeffs, T xi, T yi) const {
        T value = T(0.0);
        for (int i = 0; i < N_PSF_BASIS; ++i)
            value += exp(coeffs[i]) * basis_[i].evaluate<T> (xi, yi);
        return value;
    };

private:

    int status_;
    vector<Mixture> basis_;

};

};

#endif
