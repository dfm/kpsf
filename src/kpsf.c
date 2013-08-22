#include <Python.h>
#include <numpy/arrayobject.h>
#include "solver.h"

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#define PARSE_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, \
        NPY_IN_ARRAY)

static char solve_doc[] =
    "Solve the shizz.\n";

static PyObject
*kpsf_py_solve (PyObject *self, PyObject *args)
{
    PyObject *data_obj, *dim_obj, *coords_obj, *psf_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &data_obj, &dim_obj, &coords_obj,
                          &psf_obj))
        return NULL;

    // Parse the numpy arrays.
    PyArrayObject *data_array = PARSE_ARRAY(data_obj),
                  *dim_array = PARSE_ARRAY(dim_obj),
                  *coords_array = PARSE_ARRAY(coords_obj),
                  *psf_array = PARSE_ARRAY(psf_obj);
    if (data_array == NULL || dim_array == NULL || coords_array == NULL ||
        psf_array == NULL) {
        Py_XDECREF(data_array);
        Py_XDECREF(dim_array);
        Py_XDECREF(coords_array);
        Py_XDECREF(psf_array);
        return NULL;
    }

    // Figure out the dimensions.
    int ntime = (int) PyArray_DIM (data_array, 0),
        npixels = (int) PyArray_DIM (data_array, 1);

    // Access the arrays.
    double *data = PyArray_DATA(data_array),
           *dim = PyArray_DATA(dim_array),
           *coords = PyArray_DATA(coords_array),
           *psfpars = PyArray_DATA(psf_array);

    // Solve using Ceres.
    int info = kpsf_solve(ntime, npixels, data, dim, coords, psfpars, 1);

    // Build the output.
    PyObject *ret = Py_BuildValue ("iOO", info, coords_array, psf_array);

    // Clean up.
    Py_DECREF(data_array);
    Py_DECREF(dim_array);
    Py_DECREF(coords_array);
    Py_DECREF(psf_array);

    return ret;
}

static PyMethodDef kpsf_methods[] = {
    {"solve", (PyCFunction) kpsf_py_solve, METH_VARARGS, solve_doc},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int kpsf_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int kpsf_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_kpsf",
    NULL,
    sizeof(struct module_state),
    kpsf_methods,
    NULL,
    kpsf_traverse,
    kpsf_clear,
    NULL
};

#define INITERROR return NULL

PyObject *PyInit__kpsf(void)
#else
#define INITERROR return

void init_kpsf(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_kpsf", kpsf_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_kpsf.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
