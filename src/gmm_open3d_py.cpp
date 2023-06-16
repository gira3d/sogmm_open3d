#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <sogmm_open3d/GMM.h>

namespace py = pybind11;

template <typename T, uint32_t D>
void binding_generator(py::module& m, std::string& typestr)
{
  static constexpr uint32_t C = D * D;
  using GMMClass = GMM<T, D>;
  std::string pyclass_name = std::string("GMM") + typestr;
  py::class_<GMMClass>(m, pyclass_name.c_str(), py::dynamic_attr(),
                       py::module_local())
      .def(py::init())
      .def(py::init<unsigned int, unsigned int>())
      .def(py::init<unsigned int, unsigned int, std::string>())
      .def(py::init<unsigned int, unsigned int, std::string, bool>())
      .def(py::init<unsigned int, unsigned int, std::string, bool, std::string,
                    std::string>())
      .def_readwrite("n_components_", &GMMClass::n_components_)
      .def_readwrite("n_samples_", &GMMClass::n_samples_)
      .def_readwrite("tol_", &GMMClass::tol_)
      .def_readwrite("reg_covar_", &GMMClass::reg_covar_)
      .def_readwrite("max_iter_", &GMMClass::max_iter_)
      .def_readwrite("support_size_", &GMMClass::support_size_)
      .def_readwrite("weights_", &GMMClass::weights_)
      .def_readwrite("means_", &GMMClass::means_)
      .def_readwrite("covariances_", &GMMClass::covariances_)
      .def_readwrite("covariances_cholesky_", &GMMClass::covariances_cholesky_)
      .def_readwrite("precisions_cholesky_", &GMMClass::precisions_cholesky_)
      .def("update_device_and_host_external",
           &GMMClass::updateDeviceAndHostExternal)
      .def("sample", &GMMClass::sample)
      .def("color_conditional", &GMMClass::colorConditional)
      .def("score",
           [](GMMClass& g, const typename GMMClass::MatrixXD& X) {
             // take to the GPU
             typename GMMClass::Tensor Xt = EigenMatrixToTensor(X, g.device_);

             return g.score(Xt);
           })
      .def("score_samples",
           [](GMMClass& g, const typename GMMClass::MatrixXD& X) {
             // take to the GPU
             typename GMMClass::Tensor Xt = EigenMatrixToTensor(X, g.device_);

             // compute the scores on GPU
             typename GMMClass::Tensor output;
             g.scoreSamples(Xt, output);

             // return to CPU
             return TensorToEigenMatrix<T>(output.Reshape({ X.rows(), 1 }));
           })
      .def("fit",
           [](GMMClass& g, const typename GMMClass::MatrixXD& X,
              const typename GMMClass::Matrix& resp) {
             // take to the GPU
             typename GMMClass::Tensor Xt = EigenMatrixToTensor(X, g.device_);
             typename GMMClass::Tensor Respt =
                 EigenMatrixToTensor(resp, g.device_);

             // fit
             bool success = g.fit(Xt, Respt);

             // update CPU members
             g.updateHostfromDevice();

             return success;
           })
      .def("e_step",
           [](GMMClass& g, const typename GMMClass::MatrixXD& X) {
             // take to the GPU
             typename GMMClass::Tensor Xt = EigenMatrixToTensor(X, g.device_);

             // run eStep once
             // stores Log_Resp_ internally
             g.eStep(Xt);

             // return Log_Resp_ on CPU
             return TensorToEigenMatrix<T>(
                 g.getLogResp().Reshape({ X.rows(), g.n_components_ }));
           })
      .def("m_step",
           [](GMMClass& g, const typename GMMClass::MatrixXD& X,
              const typename GMMClass::Matrix& resp) {
             // take to the GPU
             typename GMMClass::Tensor Xt = EigenMatrixToTensor(X, g.device_);
             typename GMMClass::Tensor Respt =
                 EigenMatrixToTensor(resp, g.device_);

             // run mStep once
             // stores the output within the class object
             g.mStep(Xt, Respt);

             // copy to CPU members
             g.updateHostfromDevice();
           })
      .def(py::pickle(
          [](const GMMClass& g) {
            return py::make_tuple(
                g.n_components_, g.tol_, g.reg_covar_, g.max_iter_,
                g.support_size_, g.weights_, g.means_, g.covariances_,
                g.precisions_cholesky_, g.covariances_cholesky_);
          },
          [](py::tuple t) {
            GMMClass g;
            g.n_components_ = t[0].cast<unsigned int>();
            g.tol_ = t[1].cast<T>();
            g.reg_covar_ = t[2].cast<T>();
            g.max_iter_ = t[3].cast<unsigned int>();
            g.support_size_ = t[4].cast<unsigned int>();
            g.weights_ = t[5].cast<Eigen::Matrix<T, Eigen::Dynamic, 1>>();
            g.means_ = t[6].cast<
                Eigen::Matrix<T, Eigen::Dynamic, D,
                              (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>();
            g.covariances_ = t[7].cast<
                Eigen::Matrix<T, Eigen::Dynamic, C,
                              (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>();
            g.precisions_cholesky_ = t[8].cast<
                Eigen::Matrix<T, Eigen::Dynamic, C,
                              (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>();
            g.covariances_cholesky_ = t[9].cast<
                Eigen::Matrix<T, Eigen::Dynamic, C,
                              (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>();
            return g;
          }));
}

PYBIND11_MODULE(gmm_open3d_py, g)
{
  std::string t1 = "f1GPU";
  binding_generator<float, 1>(g, t1);

  std::string t2 = "f2GPU";
  binding_generator<float, 2>(g, t2);

  std::string t3 = "f3GPU";
  binding_generator<float, 3>(g, t3);

  std::string t4 = "f4GPU";
  binding_generator<float, 4>(g, t4);
}