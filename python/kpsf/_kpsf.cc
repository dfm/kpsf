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

static char module_doc[] = "\n";
static char photometry_doc[] = "\n";

static PyObject* kpsf_photometry (PyObject* self, PyObject* args)
{
    Py_INCREF(Py_None);
    return Py_None;
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
