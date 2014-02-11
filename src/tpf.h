#ifndef _TPF_H_
#define _TPF_H_

#include <vector>
#include <fitsio.h>
#include <Eigen/Dense>

// Ignore string warnings. It's cfitsio's fault!
#pragma GCC diagnostic ignored "-Wwrite-strings"

using std::vector;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::MatrixXi;

namespace kpsf {

//
// Read a Kepler target pixel file into a vector of images.
//
int load_tpf (const char* fn, vector<MatrixXd>* flux, vector<double>* time,
              long* ccd)
{
    fitsfile *f;
    int status = 0;

    // Open the file.
    if (fits_open_file(&f, fn, READONLY, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Get the CCD number.
    char comm[68];
    if (fits_read_key_lng(f, "OUTPUT", ccd, comm, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Move to the mask HDU.
    int hdutype;
    if (fits_movabs_hdu(f, 3, &hdutype, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Get the dimensions.
    int nfound;
    long dim[2];
    if (fits_read_keys_lng(f, "NAXIS", 1, 2, dim, &nfound, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Move to the data HDU.
    if (fits_movabs_hdu(f, 2, &hdutype, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Get the number of rows.
    long nrows;
    if (fits_read_key_lng(f, "NAXIS2", &nrows, comm, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Read the flux and uncertainty columns.
    int anynull, count = 0;
    double nullval = -1, t;
    *flux = vector<MatrixXd>(0);
    *time = vector<double>(0);
    for (int i = 0; i < 2000; ++i) {
    /* for (int i = 0; i < nrows; ++i) { */
        MatrixXd tmp(dim[0], dim[1]);
        if (fits_read_col(f, TDOUBLE, 4, i+1, 1, dim[0]*dim[1], &nullval,
                        &(tmp(0, 0)), &anynull, &status)) {
            fits_report_error(stderr, status);
            return status;
        }

        if (tmp.sum() > 0) {
            (*flux).push_back(tmp);

            if (fits_read_col(f, TDOUBLE, 1, i+1, 1, 1, &nullval, &t,
                              &anynull, &status)) {
                fits_report_error(stderr, status);
                return status;
            }
            (*time).push_back(double(t));
        }
    }

    // Close the file and clean up.
    if (fits_close_file(f, &status))
        fits_report_error(stderr, status);

    return status;
};

};

#endif
