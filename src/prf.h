#ifndef _PRF_H_
#define _PRF_H_

#include <vector>
#include <fitsio.h>
#include <Eigen/Dense>

using std::vector;
using Eigen::MatrixXd;

namespace kpsf {

//
// Constants for the Kepler PRF model. These will be different for other
// models.
//
#define N_PSF_BASIS 5
#define N_PSF_COEFF 4
#define OVERSAMPLE  50
#define DIM_X       550
#define DIM_Y       550
#define CENTER_X    275
#define CENTER_Y    275

//
// Read in the Kepler PRF basis images from a given FITS file. Returns a
// vector of images in Eigen::MatrixXd format. Status will be non-zero on
// failure.
//
int load_prfs (const char *fn, vector<MatrixXd>* prfs)
{
    fitsfile *f;
    int status = 0;

    // Open the file.
    if (fits_open_file(&f, fn, READONLY, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Read in the data.
    int anynull, hdutype, i;
    long fpixel = 1;
    double nullval = -1;
    *prfs = vector<MatrixXd>(N_PSF_BASIS);
    for (i = 0; i < N_PSF_BASIS; ++i) {
        // Move to the correct HDU.
        if (fits_movabs_hdu(f, i+2, &hdutype, &status)) {
            fits_report_error(stderr, status);
            return status;
        }

        // Allocate the matrix and read the image.
        (*prfs)[i] = MatrixXd(DIM_X, DIM_Y);
        if (fits_read_img(f, TDOUBLE, fpixel, DIM_X * DIM_Y, &nullval,
                          &((*prfs)[i](0)), &anynull, &status)) {
            fits_report_error(stderr, status);
            return status;
        }
    }

    // Close the file.
    if (fits_close_file(f, &status))
        fits_report_error(stderr, status);

    return status;
};

};

#endif
