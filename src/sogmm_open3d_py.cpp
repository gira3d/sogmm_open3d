#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <chrono>
#include <unistd.h>

#include <open3d/core/Device.h>
#include <open3d/core/CUDAUtils.h>
#include <open3d/core/linalg/Matmul.h>
#include <open3d/core/linalg/MatmulBatched.h>
#include <open3d/core/EigenConverter.h>
#include <self_organizing_gmm/TimeProfiler.h>

#include <self_organizing_gmm/GMM.h>
#include <sogmm_open3d/Open3DUtils.h>

namespace py = pybind11;
namespace o3d = open3d;
typedef o3d::core::Tensor Tensor;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrix;

void test(Eigen::MatrixXf A)
{
  // Figure out what version of SOGMM to run.
  int response = o3d::core::cuda::DeviceCount();
  if (!response)
  {
    std::cerr << "No CUDA device found; running SOGMM CPU." << std::endl;
  }
  else
  {
    std::cerr << "CUDA device found" << std::endl;

    int open3d_cuda = o3d::core::cuda::IsAvailable();
    if (!open3d_cuda)
    {
      std::cerr << "Open3D not compiled with CUDA support; running SOGMM CPU."
                << std::endl;
    }
    else
    {
      std::cerr << "Open3D compiled with CUDA support; running SOGMM GPU."
                << std::endl;
    }
  }

  // CPU v/s GPU matrix multiplication
  // convert to open3d tensors on GPU
  typedef o3d::core::Tensor Tensor;
  Tensor At = EigenMatrixToTensor(A, o3d::core::Device("CUDA:0"));
  Tensor ret = At.LLT();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c =
      TensorToEigenMatrix<float>(ret);
  std::cerr << "CUDA LLT output:\n" << c << std::endl;

  // inverse correct answer
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> I =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>::Identity(3, 3);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      crt_answer = A.llt().solve(I);
  std::cerr << "Eigen LLT inverse output: " << crt_answer << std::endl;

  // compute inverse of At using LLT solver
  Tensor Atinv =
      At.SolveLLT(Tensor::Eye(3, At.GetDtype(), o3d::core::Device("CUDA:0")));
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      at_inv_eigen = TensorToEigenMatrix<float>(Atinv);
  std::cerr << "CUDA LLT inverse output: " << at_inv_eigen << std::endl;

  Tensor Dt = EigenMatrixToTensor(A, o3d::core::Device("CPU:0"));
  Tensor ret2 = Dt.LLT();
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> e =
      TensorToEigenMatrix<float>(ret2);
  std::cerr << "CPU LLT output: " << e << std::endl;

  // compute inverse of At using LLT solver
  Tensor Dtinv =
      ret2.SolveLLT(Tensor::Eye(3, Dt.GetDtype(), o3d::core::Device("CPU:0")));
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dt_inv_eigen = TensorToEigenMatrix<float>(Dtinv);
  std::cerr << "CPU LLT inverse output: " << dt_inv_eigen << std::endl;
}

// Assumes 2D input
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
diagonal(const Eigen::MatrixXf& in)
{
  int DD = in.cols();
  int D = std::sqrt(DD);
  int K = in.rows();

  if (DD != D * D)
  {
    std::cerr << "Second dimension must be square of a number" << std::endl;
    return Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>();
  }

  Tensor mchol =
      EigenMatrixToTensor(in, o3d::core::Device("CUDA:0")).Reshape({ K, D, D });
  Tensor diagonals = GetDiagonal(mchol).Flatten(1, 1);
  return TensorToEigenMatrix<float>(diagonals);
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
computeLogDetCholesky(const Eigen::MatrixXf& in)
{
  int DD = in.cols();
  int D = std::sqrt(DD);
  int K = in.rows();

  Tensor in_Tensor =
      EigenMatrixToTensor(in, o3d::core::Device("CUDA:0")).Reshape({ K, D, D });
  Tensor diagonals = GetDiagonal(in_Tensor).Flatten(1, 1);
  return TensorToEigenMatrix<float>(((diagonals.Log()).Sum({ 1 }, true)));
}

Matrix computeCholeskyForLoop(const Eigen::MatrixXf& in)
{
  int DD = in.cols();
  int D = std::sqrt(DD);
  int K = in.rows();
  int idx = 0;

  int start_idx = 0;
  int end_idx = 1;

  Tensor inTensor =
      EigenMatrixToTensor(in, o3d::core::Device("CUDA:0")).Reshape({ K, D, D });

  /////
  // Line below is only for returning result via binding
  /////
  Matrix out = Matrix::Zero(in.rows(), in.cols());

  for (int k = 0; k < K; ++k)
  {
    // CUDA
    Tensor slice_reshaped = inTensor[k].Reshape({ D, D });
    Tensor llt = slice_reshaped.LLT();
    Tensor soln = (((llt.T()).Matmul(llt))
                       .SolveLLT((llt.T()).Matmul(Tensor::Eye(
                           D, llt.GetDtype(), o3d::core::Device("CUDA:0")))))
                      .T();

    // Binding
    Matrix tmp = TensorToEigenMatrix<float>(soln);
    out.row(k) << Eigen::Map<Matrix>(tmp.data(), 1, DD);

    // Eigen check
    typedef float T;
    using VectorC = Eigen::Matrix<T, -1, 1>;
    using MatrixDD = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;

    VectorC c = in.row(k);
    MatrixDD cov = Eigen::Map<MatrixDD>(c.data(), D, D);
    MatrixDD cov_chol = cov.llt().matrixL();
    MatrixDD prec_chol =
        (cov_chol.transpose() * cov_chol)
            .llt()
            .solve(cov_chol.transpose() * MatrixDD::Identity(D, D))
            .transpose();

    // std::cerr << "Eigen Solution\n" << prec_chol << std::endl;
    // std::cerr << "Tensor Solution\n" << soln.ToString()  << std::endl;
  }
  return out;
}

Matrix logSumExpCols(const Eigen::MatrixXf& in)
{
  int DD = in.cols();
  int K = in.rows();
  int idx = 0;

  Tensor inTensor = EigenMatrixToTensor(in, o3d::core::Device("CUDA:0"));
  Tensor amax = inTensor.Max({ 1 }, true);
  Tensor logsumexp = ((inTensor - amax).Exp()).Sum({ 1 }, true).Log();
  Tensor soln = logsumexp + amax;

  // Binding
  Matrix tmp = TensorToEigenMatrix<float>(soln);
  return tmp;
}

void logSumExp(const Tensor& in, const unsigned int dim, Tensor& out)
{
  Tensor amax = in.Max({ dim }, true);
  Tensor logsumexp = ((in - amax).Exp()).Sum({ dim }, true).Log();

  out = logsumexp + amax;
}

Tensor scoreSamples(const Tensor& Xt, const Tensor& Means,
                    const Tensor& Weights, const Tensor& Precs_Chol)
{
  int N = Xt.GetShape()[0];
  int D = Xt.GetShape()[1];
  int C = D * D;
  int n_components = Means.GetShape()[1];
  int n_samples = N;

  // Term 2 of Equation (3.14)
  Tensor Log_Det_Cholesky =
      ((GetDiagonal(Precs_Chol[0]).Log()).Sum({ 1 }, true))
          .Reshape({ 1, n_components, 1 });

  // Diff, PDiff, Log_Gaussian_Prob are all terms for the first
  // term in Equation (3.14)
  Tensor Diff = (Xt.Reshape({ n_samples, 1, D }) - Means)
                    .Reshape({ n_samples, n_components, D, 1 });

  Tensor PDiff = (Precs_Chol.Mul(Diff))
                     .Sum({ 2 }, true)
                     .Reshape({ n_samples, n_components, D, 1 });

  // Equation (3.14), output of estimateLogGaussianProb
  Tensor Log_Gaussian_Prob =
      ((((PDiff.Mul(PDiff)).Sum({ 2 }, false)).Add(D * LOG_2_M_PI)).Mul(-0.5))
          .Add(Log_Det_Cholesky);

  // This is the first two terms in Equation (3.7)
  Tensor WeightedLogProb = Log_Gaussian_Prob.Add(Weights.Log());

  // Log likelihood for each sample
  Tensor PerSampleLogLikelihood;
  logSumExp(WeightedLogProb, 1, PerSampleLogLikelihood);
  return PerSampleLogLikelihood;
}

///////////////////
// X \in \mathbb{R}^{NxD}
// means \in \mathbb{R}^{KxD}
// weights \in \mathbb{R}^{Kx1}
// precisions_chol \in \mathbb{R}^{KxC}
//
// where N = number of points
//       D = dimension of data
//       C = D * D
///////////////////
Matrix scoreSamplesWrapper(const Eigen::MatrixXf& X,
                           const Eigen::MatrixXf& means,
                           const Eigen::MatrixXf& weights,
                           const Eigen::MatrixXf& precisions_chol)
{
  TimeProfiler tp;
  int D = X.cols();
  int N = X.rows();
  int C = D * D;
  int n_components = means.rows();
  Dtype dtype = Dtype::FromType<float>();

  Device device = o3d::core::Device("CUDA:0");

  Tensor Means = Tensor::Zeros({ 1, n_components, D }, dtype, device);
  Tensor Weights = Tensor::Zeros({ 1, n_components, 1 }, dtype, device);
  Tensor Precs_Chol = Tensor::Zeros({ 1, n_components, D, D }, dtype, device);

  Tensor Xt = EigenMatrixToTensor(X, device);
  Means = EigenMatrixToTensor(means, device).Reshape(Means.GetShape());
  Weights = EigenMatrixToTensor(weights, device).Reshape(Weights.GetShape());
  Precs_Chol = EigenMatrixToTensor(precisions_chol, device)
                   .Reshape(Precs_Chol.GetShape());

  tp.tic("score_samples");
  Tensor PerSampleLogLikelihood = scoreSamples(Xt, Means, Weights, Precs_Chol);
  auto duration = tp.toc("score_samples");
  std::cerr << "Time: " << duration << std::endl;

  // Binding
  return TensorToEigenMatrix<float>(PerSampleLogLikelihood.Reshape({ N, 1 }));
}

Matrix matMulBatched(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B)
{
  TimeProfiler tp;
  int DD = A.cols();
  int D = std::sqrt(DD);  // DD needs to be a perfect square.
  int K = A.rows();

  Tensor ATensorCUDA =
      EigenMatrixToTensor(A, o3d::core::Device("CUDA:0")).Reshape({ K, D, D });
  Tensor BTensorCUDA =
      EigenMatrixToTensor(B, o3d::core::Device("CUDA:0")).Reshape({ K, D, D });

  tp.tic("CUDAMatmulBatched");
  Tensor CTensor = ATensorCUDA.MatmulBatched(BTensorCUDA);
  auto cuda_mat_mul_time = tp.toc("CUDAMatmulBatched");
  std::cerr << "cuda mat mul 3d time " << cuda_mat_mul_time << std::endl;

  return TensorToEigenMatrix<float>(CTensor.Reshape({ K, DD }));
}

Matrix
sum(const Eigen::MatrixXf& A)
{
  TimeProfiler tp;
  int DD = A.cols();
  int D = std::sqrt(DD);  // DD needs to be a perfect square.
  int K = A.rows();

  Tensor ATensorCUDA =
      EigenMatrixToTensor(A, o3d::core::Device("CUDA:0")).Reshape({ K, D, D });

  tp.tic("Sum");
  Tensor CTensor = ATensorCUDA.Sum({1}, true);
  auto time = tp.toc("Sum");
  std::cerr << "cuda sum: " << time << std::endl;

  std::cerr << "CTensor: " << CTensor.GetShape().ToString() << std::endl;
  return TensorToEigenMatrix<float>(CTensor.Reshape({ K, D }));
}

Matrix
sum_(const Eigen::MatrixXf& A)
{
  TimeProfiler tp;
  int DD = A.cols();
  int D = std::sqrt(DD);  // DD needs to be a perfect square.
  int K = A.rows();

  Tensor ATensorCUDA =
      EigenMatrixToTensor(A, o3d::core::Device("CUDA:0")).Reshape({ K, D, D });
  Tensor BTensorCUDA = Tensor::Zeros({K, 1, D}, ATensorCUDA.GetDtype(), ATensorCUDA.GetDevice());

  tp.tic("Sum_");
  ATensorCUDA.Sum_({1}, true, BTensorCUDA);
  auto time = tp.toc("Sum_");
  std::cerr << "cuda sum_: " << time << std::endl;

  std::cerr << "BTensorCUDA: " << BTensorCUDA.GetShape().ToString() << std::endl;

  return TensorToEigenMatrix<float>(BTensorCUDA.Reshape({ K, D }));
}


PYBIND11_MODULE(sogmm_open3d_py, g)
{
  g.def("test", &test);
  g.def("diagonal", &diagonal);
  g.def("compute_log_det_cholesky", &computeLogDetCholesky);
  g.def("compute_cholesky_for_loop", &computeCholeskyForLoop);
  g.def("log_sum_exp_cols", &logSumExpCols);
  g.def("mat_mul_batched", &matMulBatched);
  g.def("score_samples", &scoreSamplesWrapper);
  g.def("sum", &sum);
  g.def("sum_", &sum_);
}
