#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "csb.h"

namespace py = pybind11;

PYBIND11_MODULE(csb_pybind, m) {
    py::class_<World>(m, "World")
        .def(py::init<>());
}
