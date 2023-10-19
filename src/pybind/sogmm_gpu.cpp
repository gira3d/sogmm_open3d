#include <pybind11/pybind11.h>

#include <sogmm_open3d/pybind/SOGMM.h>

namespace py = pybind11;

PYBIND11_MODULE(sogmm_gpu, g)
{
  std::string t2 = "f2Device";
  std::string t3 = "f3Device";
  std::string t4 = "f4Device";
  sogmm::gpu::container_binding_generator<float, 2>(g, t2);
  sogmm::gpu::container_binding_generator<float, 3>(g, t3);
  sogmm::gpu::container_binding_generator<float, 4>(g, t4);

  std::string f = "Learner";
  sogmm::gpu::learner_binding_generator<float>(g, f);

  f = "Inference";
  sogmm::gpu::inference_binding_generator<float>(g, f);

  g.def("marginal_X", &sogmm::gpu::extractXpart<float>);
}