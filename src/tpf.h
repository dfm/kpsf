#ifndef _TPF_H_
#define _TPF_H_

#include <vector>
#include <fitsio.h>
#include <Eigen/Dense>

using std::vector;
using Eigen::MatrixXd;
using Eigen::MatrixXi;

namespace kpsf {

//
// Read a Kepler target pixel file into a vector of images.
//
int load_tpf (const char *fn, vector<MatrixXd>* flux, vector<MatrixXd>* ferr)
{
    fitsfile *f;
    int status = 0;

    // Open the file.
    if (fits_open_file(&f, fn, READONLY, &status)) {
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
    char comm[68];
    long nrows;
    if (fits_read_key_lng(f, "NAXIS2", &nrows, comm, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Read the flux and uncertainty columns.
    int anynull;
    double nullval = -1;
    *flux = vector<MatrixXd>(nrows);
    *ferr = vector<MatrixXd>(nrows);
    for (int i = 0; i < nrows; ++i) {
        (*flux)[i] = MatrixXd(dim[0], dim[1]);
        if (fits_read_col(f, TDOUBLE, 5, i+1, 1, dim[0]*dim[1], &nullval,
                          &((*flux)[i](0, 0)), &anynull, &status)) {
            fits_report_error(stderr, status);
            return status;
        }

        (*ferr)[i] = MatrixXd(dim[0], dim[1]);
        if (fits_read_col(f, TDOUBLE, 6, i+1, 1, dim[0]*dim[1], &nullval,
                          &((*ferr)[i](0, 0)), &anynull, &status)) {
            fits_report_error(stderr, status);
            return status;
        }
    }

    // Close the file and clean up.
    if (fits_close_file(f, &status))
        fits_report_error(stderr, status);

    return status;
};

};

#endif
