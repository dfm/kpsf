#include <vector>
#include <iostream>
#include <fitsio.h>
#include <Eigen/Dense>

#include "tpf.h"
#include "kpsf.h"

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using kpsf::load_tpf;
using kpsf::photometry;


int write_results (const char* fn, vector<double> time, MatrixXd flat,
                   vector<double> coords, double bg, double* coeffs)
{
    fitsfile *f;
    int status = 0, nx = flat.rows(), ny = flat.cols(), nt = coords.size();

    // Create the FITS file.
    remove(fn);
    if (fits_create_file(&f, fn, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Create a new binary table.
    char extname[] = "results",
        * ttype[] = {"time", "flux", "xpos", "ypos"},
        * tform[] = {"1D",   "1D",   "1D",   "1D"},
        * tunit[] = {"KBJD", "\0",   "pix",  "pix"};
    if (fits_create_tbl(f, BINARY_TBL, nt, 4, ttype, tform, tunit, extname,
                        &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Write the columns.
    for (int i = 0; i < nt; ++i) {
        fits_write_col(f, TDOUBLE, 1, i+1, 1, 1, &(time[i]), &status);
        fits_write_col(f, TDOUBLE, 2, i+1, 1, 1, &(coords[3*i]),   &status);
        fits_write_col(f, TDOUBLE, 3, i+1, 1, 1, &(coords[3*i+1]), &status);
        fits_write_col(f, TDOUBLE, 4, i+1, 1, 1, &(coords[3*i+2]), &status);
        if (status) {
            fits_report_error(stderr, status);
            return status;
        }
    }
    // Update the with the coefficients.
    for (int i = 0; i < N_PSF_BASIS; ++i) {
        char k[100];
        sprintf(k, "COEFF%d", i);
        if (fits_update_key(f, TDOUBLE, k, &(coeffs[i]),
                            "PSF basis coefficient", &status)) {
            fits_report_error(stderr, status);
            return status;
        }
    }

    if (fits_update_key(f, TDOUBLE, "BACKGROUND", &bg, "Background level",
                        &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Create the flat image.
    long int naxes[] = {nx, ny};
    if (fits_create_img(f, DOUBLE_IMG, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Write the image.
    if (fits_write_img(f, TDOUBLE, 1, nx*ny, &(flat(0, 0)), &status)) {
        fits_report_error(stderr, status);
        return status;
    }

    // Clean up.
    if (fits_close_file(f, &status)) fits_report_error(stderr, status);

    return status;
}


int main (int argc, char **argv)
{
    // Check the command line arguments.
    if (argc != 4) {
        std::cerr << "Incorrect number of command line arguments\n";
        std::cerr << "    Usage: " << argv[0]
                  << " /path/to/psf.mog.fits"
                  << " /path/to/target_pixels.fits.gz"
                  << " /path/to/output.fits\n";
        return -1;
    }

    // Load the Target Pixel file.
    int status;
    long ccd;
    vector<MatrixXd> flux;
    vector<double> time;
    status = load_tpf(argv[2], &flux, &time, &ccd);
    if (status) return status;

    // Load the PSF basis.
    MixtureBasis* basis = new MixtureBasis(argv[1]);

    // Allocate the parameter lists.
    int nt = flux.size(),
        nx = flux[0].rows(),
        ny = flux[0].cols();
    MatrixXd flat = MatrixXd::Ones(nx, ny);
    vector<double> coords(nt*3);
    double coeffs[] = {0.2, 0.2, 0.2, 0.2, 0.2},
           bg = 0.0;

    // Initialize the coordinates.
    double w;
    for (int t = 0; t < nt; ++t) {
        coords[3*t] = flux[t].sum();
        coords[3*t+1] = 0.0;
        coords[3*t+2] = 0.0;

        // Compute center of mass.
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                w = flux[t](i, j);
                if (w > 0.0) {
                    coords[3*t+1] += i * w;
                    coords[3*t+2] += j * w;
                }
            }
        }

        coords[3*t+1] /= coords[3*t];
        coords[3*t+2] /= coords[3*t];
    }

    // Convert the images to the correct format.
    int ntot = nt*nx*ny;
    vector<double> flux_imgs(ntot),
                   ferr_imgs(ntot);

    for (int t = 0; t < nt; ++t) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                flux_imgs[(t*nx+i)*ny+j] = flux[t](i, j);
                ferr_imgs[(t*nx+i)*ny+j] = 1.0 / sqrt(flux[t](i, j));
            }
        }
    }

    // Do the photometry.
    photometry (500, basis, 30.0, 0.01, 0.01, 0.01,
                nt, nx, ny, &(flux_imgs[0]), &(ferr_imgs[0]),
                coeffs, &(coords[0]), &(flat(0, 0)), &bg);

    // Save the results.
    status = write_results (argv[3], time, flat, coords, bg, coeffs);

    delete basis;
    return status;
}
