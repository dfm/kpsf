#include <Python.h>
#include <numpy/arrayobject.h>

#include "kpsf.h"

extern "C" {
#if PY_MAJOR_VERSION >= 3
static int kpsf_traverse(PyObject* m, visitproc visit, void* arg);
static struct PyModuleDef moduledef;
static int kpsf_clear(PyObject* m);
PyObject* PyInit__kpsf (void);
#else
void init_kpsf (void);
#endif
static PyObject* kpsf_photometry (PyObject* self, PyObject* args);
}

#define PARSE_IN_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, NPY_IN_ARRAY)
#define PARSE_INOUT_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, NPY_INOUT_ARRAY)

static char module_doc[] = "\n";
static char photometry_doc[] = "\n";

static PyObject* kpsf_photometry (PyObject* self, PyObject* args)
{
    int nt, npix, maxiter;
    const char* prf_fn;
    MixtureBasis* basis;
    double loss_scale, sum_to_one_strength, psf_l2_strength, flat_reg_strength;
    PyObject* x_obj      = NULL,
            * y_obj      = NULL,
            * flux_obj   = NULL,
            * ferr_obj   = NULL,
            * coeffs_obj = NULL,
            * coords_obj = NULL,
            * ff_obj     = NULL,
            * bg_obj     = NULL;
    PyArrayObject* x_array      = NULL,
                 * y_array      = NULL,
                 * flux_array   = NULL,
                 * ferr_array   = NULL,
                 * coeffs_array = NULL,
                 * coords_array = NULL,
                 * ff_array     = NULL,
                 * bg_array     = NULL;
    double* flux_imgs, * ferr_imgs, * coeffs, * coords, * ff, * bg;
    int* x, * y;

    // Parse the input arguments.
    if (!PyArg_ParseTuple(args, "isddddOOOOOOOO", &maxiter, &prf_fn,
                          &loss_scale, &sum_to_one_strength, &psf_l2_strength,
                          &flat_reg_strength,
                          &x_obj, &y_obj, &flux_obj, &ferr_obj,
                          &coeffs_obj, &coords_obj, &ff_obj, &bg_obj))
        return NULL;

    // Load the PSF basis.
    basis = new MixtureBasis (prf_fn);
    if (basis->get_status()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Failed to load MOG PSF basis file.");
        delete basis;
        return NULL;
    }

    // Parse the arrays.
    x_array      = (PyArrayObject*) PyArray_FROM_OTF(x_obj, NPY_INT, NPY_IN_ARRAY);
    y_array      = (PyArrayObject*) PyArray_FROM_OTF(y_obj, NPY_INT, NPY_IN_ARRAY);
    flux_array   = PARSE_IN_ARRAY(flux_obj),
    ferr_array   = PARSE_IN_ARRAY(ferr_obj),
    coeffs_array = PARSE_INOUT_ARRAY(coeffs_obj),
    coords_array = PARSE_INOUT_ARRAY(coords_obj),
    ff_array     = PARSE_INOUT_ARRAY(ff_obj),
    bg_array     = PARSE_INOUT_ARRAY(bg_obj);
    if (x_array == NULL || y_array == NULL || flux_array == NULL ||
            ferr_array == NULL || coeffs_array == NULL ||
            coords_array == NULL || ff_array == NULL || bg_array == NULL)
        goto fail;

    // Basic dimension checks.
    if (PyArray_NDIM (coords_array) != 2 || PyArray_NDIM (ff_array) != 1 ||
            PyArray_NDIM (flux_array) != 2 || PyArray_NDIM (ferr_array) != 2 ||
            PyArray_DIM (coords_array, 1) != 3 ||
            PyArray_DIM (coeffs_array, 0) != N_PSF_BASIS) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        goto fail;
    }

    // Get the relevant dimensions.
    nt = PyArray_DIM (flux_array, 0);
    npix = PyArray_DIM (flux_array, 1);

    // Check these dimensions.
    if (nt != PyArray_DIM(ferr_array, 0) || npix != PyArray_DIM(ferr_array, 1) ||
            npix != PyArray_DIM(ff_array, 0) || nt != PyArray_DIM(coords_array, 0)) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        goto fail;
    }

    // Access the data.
    x         = (int*) PyArray_DATA(x_array);
    y         = (int*) PyArray_DATA(y_array);
    flux_imgs = (double*) PyArray_DATA(flux_array);
    ferr_imgs = (double*) PyArray_DATA(ferr_array);
    coeffs = (double*) PyArray_DATA(coeffs_array);
    coords = (double*) PyArray_DATA(coords_array);
    ff = (double*) PyArray_DATA(ff_array);
    bg = (double*) PyArray_DATA(bg_array);

    kpsf::photometry (maxiter, basis, loss_scale, sum_to_one_strength,
                      psf_l2_strength, flat_reg_strength, nt, npix,
                      x, y, flux_imgs, ferr_imgs, coeffs, coords, ff, bg);

    // Clean up.
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(flux_array);
    Py_DECREF(ferr_array);
    Py_DECREF(coeffs_array);
    Py_DECREF(coords_array);
    Py_DECREF(ff_array);
    Py_DECREF(bg_array);
    delete basis;

    Py_INCREF(Py_None);
    return Py_None;

fail:

    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    Py_XDECREF(flux_array);
    Py_XDECREF(ferr_array);
    Py_XDECREF(coeffs_array);
    Py_XDECREF(coords_array);
    Py_XDECREF(ff_array);
    Py_XDECREF(bg_array);
    delete basis;
    return NULL;
}

static PyMethodDef kpsf_methods[] = {
    {"photometry", kpsf_photometry, METH_VARARGS, photometry_doc},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int kpsf_traverse(PyObject* m, visitproc visit, void* arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int kpsf_clear(PyObject* m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_kpsf",
    doc,
    sizeof(struct module_state),
    kpsf_methods,
    NULL,
    kpsf_traverse,
    kpsf_clear,
    NULL
};

#define INITERROR return NULL

PyObject* PyInit__kpsf (void)
#else

#define INITERROR return

void init_kpsf (void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject* module = PyModule_Create(&moduledef);
#else
    PyObject* module = Py_InitModule3("_kpsf", kpsf_methods, module_doc);
#endif

    if (module == NULL) INITERROR;
    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
