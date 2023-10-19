#include <sogmm_open3d/SOGMMLearner.h>
#include <sogmm_open3d/SOGMMInference.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <self_organizing_gmm/pybind/SOGMM.h>

namespace py = pybind11;

namespace sogmm
{
  namespace gpu
  {
    template <typename T, uint32_t D>
    void container_binding_generator(py::module &m, std::string &typestr)
    {
      using Container = sogmm::gpu::SOGMM<T, D>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      py::class_<Container>(m, pyclass_name.c_str(), "GMM parameters container on the GPU.",
                            py::dynamic_attr())
          .def(py::init(), "Default empty constructor.")
          .def(py::init<const Container &>(),
               "Copy from an existing container.",
               py::arg("that"))
          .def(py::init<const uint32_t &>(),
               "Initialize zero members for the given number of components.",
               py::arg("n_components"))
          .def_readwrite("n_components_", &Container::n_components_,
                         "Number of components in this GMM.")
          .def_readwrite("support_size_", &Container::support_size_,
                         "Number of points in the support of this GMM.")
          .def("merge", &Container::merge,
               "Merge another container into this container.")
          .def("to_host", &Container::toHost,
               "Return this container on the host machine (CPU).")
          .def("from_host", &Container::fromHost,
               "Take a host container to the device (GPU).");
    }

    template <typename T>
    void learner_binding_generator(py::module &m, std::string &typestr)
    {
      using Learner = sogmm::gpu::SOGMMLearner<T>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      py::class_<Learner>(m, pyclass_name.c_str(), "GMM parameters container on the CPU.",
                          py::dynamic_attr())
          .def(py::init(), "Default empty constructor.")
          .def(py::init<const float &>(), "Initialize using the bandwidth parameter")
          .def("fit", &Learner::fit)
          .def("fit_em", &Learner::fit_em);
    }

    template <typename T>
    void inference_binding_generator(py::module &m, std::string &typestr)
    {
      using Container = sogmm::gpu::SOGMM<T, 4>;
      using Inference = sogmm::gpu::SOGMMInference<T>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      py::class_<Inference>(m, pyclass_name.c_str(), "GMM parameters container on the CPU.",
                            py::dynamic_attr())
          .def(py::init(), "Default empty constructor.")
          .def("reconstruct", &Inference::reconstruct)
          .def("score_3d", &Inference::score3D)
          .def("score_4d", &Inference::score4D)
          .def("generate_pcld_4d", &Inference::generatePointCloud4D)
          .def("generate_pcld_3d", &Inference::generatePointCloud3D);
    }
  }
}
