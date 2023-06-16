#pragma once

#define BUILD_CUDA_MODULE

#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Dense>
#include <gsl/gsl_randist.h>

#include <open3d/Open3D.h>
#include <open3d/core/CUDAUtils.h>
#include <sogmm_open3d/Open3DUtils.h>
#include <self_organizing_gmm/TimeProfiler.h>

#define LOG_2_M_PI 1.83788

namespace o3d = open3d;

// T -- datatypes (usually float or double)
// D -- dimension of the data
template <typename T, uint32_t D>
class GMM
{
public:
  static constexpr uint32_t C = D * D;
  static constexpr T MVN_NORM_D =
      static_cast<T>(1.0 / std::pow(std::sqrt(2.0 * M_PI), D));
  static constexpr T MVN_NORM_3 =
      static_cast<T>(1.0 / std::pow(std::sqrt(2.0 * M_PI), 3));

  using Ptr = std::shared_ptr<GMM<T, D>>;
  using ConstPtr = std::shared_ptr<const GMM<T, D>>;

  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using VectorD = Eigen::Matrix<T, D, 1>;
  using VectorC = Eigen::Matrix<T, C, 1>;

  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                               (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
  using MatrixXD = Eigen::Matrix<T, Eigen::Dynamic, D,
                                 (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
  using MatrixDX = Eigen::Matrix<T, D, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXC = Eigen::Matrix<T, Eigen::Dynamic, C,
                                 (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
  using MatrixDD = Eigen::Matrix<T, D, D, Eigen::RowMajor>;

  using Tensor = o3d::core::Tensor;
  using SizeVector = o3d::core::SizeVector;
  using Device = o3d::core::Device;
  using Dtype = o3d::core::Dtype;

  GMM() : GMM(1, 1, "CUDA:0", false, "gmm_gpu_stats", "stats.csv")
  {
  }

  GMM(unsigned int n_components, unsigned int n_samples)
    : GMM(n_components, n_samples, "CUDA:0", false, "gmm_gpu_stats",
          "stats.csv")
  {
  }

  GMM(unsigned int n_components, unsigned int n_samples, std::string device)
    : GMM(n_components, n_samples, device, false, "gmm_gpu_stats", "stats.csv")
  {
  }

  GMM(unsigned int n_components, unsigned int n_samples, std::string device,
      bool save_stats)
    : GMM(n_components, n_samples, device, save_stats, "gmm_gpu_stats",
          "stats.csv")
  {
  }

  GMM(unsigned int n_components, unsigned int n_samples, std::string device,
      const bool save_stats, const std::string& stats_dir,
      const std::string& stats_file)
  {
    n_components_ = n_components;
    n_samples_ = n_samples;
    tol_ = 1e-3;
    reg_covar_ = 1e-6;
    max_iter_ = 100;

    device_ = o3d::core::Device(device);
    dtype_ = Dtype::FromType<T>();

    initialize(save_stats, stats_dir, stats_file);
  }

  ~GMM()
  {
  }

  void initialize(const bool save_stats, const std::string& stats_dir,
                  const std::string& stats_file)
  {
    tp_ = TimeProfiler();
    if (save_stats)
    {
      tp_.save(stats_dir, stats_file);
    }

    int N = n_samples_;
    int K = n_components_;

    Weights_ = Tensor::Zeros({ 1, K, 1 }, dtype_, device_);
    Means_ = Tensor::Zeros({ 1, K, D }, dtype_, device_);
    Covariances_ = Tensor::Zeros({ 1, K, D, D }, dtype_, device_);
    Covs_Chol_ = Tensor::Zeros({ 1, K, D, D }, dtype_, device_);
    Precs_Chol_ = Tensor::Zeros({ 1, K, D, D }, dtype_, device_);

    amax__ = Tensor::Zeros({ N, 1, 1 }, dtype_, device_);
    Log_Det_Cholesky__ = Tensor::Zeros({ K, 1 }, dtype_, device_);
    Log_Det_Cholesky_Tmp__ = Tensor::Zeros({ K, D }, dtype_, device_);
    eDiff__ = Tensor::Zeros({ N, K, D, 1 }, dtype_, device_);
    Log_Prob_Norm__ = Tensor::Zeros({ N, 1, 1 }, dtype_, device_);

    Nk__ = Tensor::Zeros({ 1, K }, dtype_, device_);
    mDiff__ = Tensor::Zeros({ K, D, N }, dtype_, device_);
    third_eye__ = Tensor::Zeros({ K, D, D }, dtype_, device_);
    A__ = Tensor::Zeros({ K, D, D }, dtype_, device_);
    B__ = Tensor::Zeros({ K, D, D }, dtype_, device_);
    Log_Resp__ = Tensor::Zeros({ N, K, 1 }, dtype_, device_);

    updateHostfromDevice();

    gsl_rng_env_setup();
    r_global_ = gsl_rng_alloc(gsl_rng_default);
    normal_dist_ =
        std::normal_distribution<T>(static_cast<T>(0.0), static_cast<T>(1.0));
  }

  void updateHostfromDevice()
  {
    weights_ = TensorToEigenMatrix<T>(Weights_.Reshape({ n_components_, 1 }));
    means_ = TensorToEigenMatrix<T>(Means_.Reshape({ n_components_, D }));
    covariances_ =
        TensorToEigenMatrix<T>(Covariances_.Reshape({ n_components_, D * D }));
    covariances_cholesky_ =
        TensorToEigenMatrix<T>(Covs_Chol_.Reshape({ n_components_, D * D }));
    precisions_cholesky_ =
        TensorToEigenMatrix<T>(Precs_Chol_.Reshape({ n_components_, D * D }));
  }

  void updateDevicefromHost()
  {
    Weights_ =
        EigenMatrixToTensor(weights_, device_).Reshape(Weights_.GetShape());
    Means_ = EigenMatrixToTensor(means_, device_).Reshape(Means_.GetShape());
    Covariances_ = EigenMatrixToTensor(covariances_, device_)
                       .Reshape(Covariances_.GetShape());
    Covs_Chol_ = EigenMatrixToTensor(covariances_cholesky_, device_)
                     .Reshape(Covs_Chol_.GetShape());
    Precs_Chol_ = EigenMatrixToTensor(precisions_cholesky_, device_)
                      .Reshape(Precs_Chol_.GetShape());
  }

  void updateDeviceAndHostExternal(const Vector& weights, const MatrixXD& means,
                                   const MatrixXC& covariances,
                                   const MatrixXC& precisions_cholesky)
  {
    Weights_ =
        EigenMatrixToTensor(weights, device_).Reshape(Weights_.GetShape());
    Means_ = EigenMatrixToTensor(means, device_).Reshape(Means_.GetShape());
    Covariances_ = EigenMatrixToTensor(covariances, device_)
                       .Reshape(Covariances_.GetShape());
    Covs_Chol_ = Covariances_[0].LLTBatched().Reshape(Covs_Chol_.GetShape());
    Precs_Chol_ = EigenMatrixToTensor(precisions_cholesky, device_)
                      .Reshape(Precs_Chol_.GetShape());
  }

  void logSumExp(const Tensor& in, const unsigned int dim, Tensor& out)
  {
    amax__ = in.Max({ dim }, true);
    ((in - amax__).Exp()).Sum_({ dim }, true, out);
    out.Log_();
    out.Add_(amax__);
  }

  void estimateWeightedLogProb(const Tensor& Xt, unsigned int n_samples,
                               Tensor& Weighted_Log_Prob)
  {
    // Term 2 of Equation (3.14)
    //Log_Det_Cholesky__ = ((GetDiagonal(Precs_Chol_[0]).Log()).Sum({ 1 }, true))
    //                         .Reshape({ 1, n_components_, 1 });

    int N = n_samples_;
    int K = n_components_;

    Log_Det_Cholesky__.Fill<float>(0.0);
    Tensor Log_Det_Cholesky_View = (GetDiagonal(Precs_Chol_[0]));
    Log_Det_Cholesky_Tmp__.CopyFrom(Log_Det_Cholesky_View);
    Log_Det_Cholesky_Tmp__.Log_();
    Log_Det_Cholesky_Tmp__.Sum_({ 1 }, true, Log_Det_Cholesky__);

    // Diff, PDiff, Log_Gaussian_Prob are all terms for the first
    // term in Equation (3.14)
    eDiff__ = (Xt.Reshape({ n_samples, 1, D }) - Means_)
                  .Reshape({ n_samples, n_components_, D, 1 });

    eDiff__ = (Precs_Chol_.Mul(eDiff__))
                  .Sum({ 2 }, true)
                  .Reshape({ n_samples, n_components_, D, 1 });

    // Equation (3.14), output of estimateLogGaussianProb
    (eDiff__.Mul(eDiff__)).Sum_({ 2 }, false, Weighted_Log_Prob);
    Weighted_Log_Prob.Add_(D * LOG_2_M_PI);
    Weighted_Log_Prob.Mul_(-0.5);
    Weighted_Log_Prob.Add_(Log_Det_Cholesky__.Reshape({ 1, K, 1 }));

    // This is the first two terms in Equation (3.7)
    Weighted_Log_Prob.Add_(Weights_.Log());
  }

  void eStep(const Tensor& Xt)
  {
    tp_.tic("eStep");

    if (n_samples_ == 0)
    {
      SizeVector Xt_shape = Xt.GetShape();
      n_samples_ = Xt_shape[0];
    }

    estimateWeightedLogProb(Xt, n_samples_, Log_Resp__);

    Log_Prob_Norm__.Fill<float>(0.0);
    logSumExp(Log_Resp__, 1, Log_Prob_Norm__);

    Log_Resp__.Sub_(Log_Prob_Norm__);

    likelihood_ = Log_Prob_Norm__.Mean({ 0, 1, 2 }, false).Item<T>();

    tp_.toc("eStep");
  }

  void mStep(const Tensor& Xt, const Tensor& Respt)
  {
    tp_.tic("mStep");
    if (n_samples_ == 0)
    {
      SizeVector Xt_shape = Xt.GetShape();
      n_samples_ = Xt_shape[0];
    }

    // initialize tensor for weights
    Nk__.Fill<float>(0.0);
    Respt.Sum_({ 0 }, true, Nk__);
    Weights_ = Nk__.T();
    Weights_.Add_(Tensor::Ones({ n_components_, 1 }, dtype_, device_) * 10 *
		  std::numeric_limits<T>::epsilon());

    // update means
    Means_ = (Respt.T().Matmul(Xt)).Div(Weights_).Reshape(Means_.GetShape());

    // update covariances
    mDiff__ = (Xt.Reshape({ n_samples_, 1, D }) - Means_)
                  .AsStrided({ n_components_, D, n_samples_ },
                             { D, 1, D * n_components_ });

    Covariances_ =
        mDiff__
            .MatmulBatched(mDiff__.Transpose(1, 2) *
                           Respt.AsStrided({ n_components_, n_samples_, 1 },
                                           { 1, n_components_, 1 }))
            .Reshape(Covariances_.GetShape());

    Covariances_.Div_(Weights_.Reshape({ 1, n_components_, 1, 1 }));

    // add reg_covar_ along the diagonal for Covariances_
    Covariances_[0]
        .AsStrided({ n_components_, D }, { C, D + 1 })
        .Add_(reg_covar_);

    // update weights
    Weights_.Div_(n_samples_);
    Weights_.Div_(Weights_.Sum({ 0 }, false));

    // update precision and covariance cholesky
    Covs_Chol_ = Covariances_[0].LLTBatched().Reshape(Covs_Chol_.GetShape());

    third_eye__.AsStrided({ n_components_, D }, { C, D + 1 })
        .Fill(static_cast<T>(1));

    A__ = Covs_Chol_[0].Transpose(1, 2).MatmulBatched(Covs_Chol_[0]);
    B__ = Covs_Chol_[0].Transpose(1, 2).MatmulBatched(third_eye__);

    Precs_Chol_ = A__.SolveLLTBatched(B__).Reshape(Precs_Chol_.GetShape());

    tp_.toc("mStep");
  }

  bool fit(const Tensor& Xt, const Tensor& Respt)
  {
    tp_.tic("fit");

    SizeVector Xt_shape = Xt.GetShape();
    n_samples_ = Xt_shape[0];

    if (n_samples_ <= 1)
    {
      throw std::runtime_error("fit: number of samples is " +
                               std::to_string(n_samples_) +
                               ", it should be greater than 1.");
    }

    if (n_components_ <= 1)
    {
      throw std::runtime_error("fit: number of components is " +
                               std::to_string(n_components_) +
                               ", it should be greater than 1.");
    }

    mStep(Xt, Respt);

    T lower_bound = -std::numeric_limits<T>::infinity();
    for (unsigned int n_iter = 0; n_iter <= max_iter_; n_iter++)
    {
      T prev_lower_bound = lower_bound;

      // E step
      eStep(Xt);

      // M step
      mStep(Xt, Log_Resp__.Exp().Reshape({ n_samples_, n_components_ }));

      // convergence check
      lower_bound = likelihood_;
      T change = lower_bound - prev_lower_bound;
      if (!std::isinf(change) && std::abs(change) < tol_)
      {
        converged_ = true;
        break;
      }
    }
    
    tp_.toc("fit");

    o3d::core::MemoryManagerCached::ReleaseCache(device_);

    if (converged_)
    {
      support_size_ = n_samples_;
      return true;
    }
    else
    {
      return false;
    }
  }

  void scoreSamples(const Tensor& Xt, Tensor& Per_Sample_Log_Likelihood)
  {
    unsigned int n_samples = Xt.GetShape()[0];

    Tensor Weighted_Log_Prob;
    estimateWeightedLogProb(Xt, n_samples, Weighted_Log_Prob);

    // Log likelihood for each sample
    logSumExp(Weighted_Log_Prob, 1, Per_Sample_Log_Likelihood);
  }

  T score(const Tensor& Xt)
  {
    Tensor PSLL;
    scoreSamples(Xt, PSLL);

    return PSLL.Mean({ 0, 1, 2 }).Item<T>();
  }

  MatrixXD sample(const unsigned int& n_samples, double sigma = 3.0)
  {
    unsigned int n_samples_comp[n_components_];

    // gsl ran multinomial only uses double datatype
    // convert the datatype of the weights vector to double
    double probs[n_components_];
    for (unsigned int i = 0; i < n_components_; i++)
    {
      probs[i] = static_cast<double>(weights_(i));
    }

    gsl_ran_multinomial(r_global_, static_cast<size_t>(n_components_),
                        n_samples, probs, n_samples_comp);

    // prepare the samples matrix
    std::vector<T> x;
    x.reserve(n_samples * D);

    while (x.size() < n_samples * D)
    {
      T rand_val = normal_dist_(generator_);

      if (std::abs(rand_val) < sigma)
      {
        x.push_back(rand_val);
      }
    }

    MatrixXD samples = Eigen::Map<MatrixXD>(x.data(), n_samples, D);

    unsigned int prev_idx = 0;
    for (unsigned int k = 0; k < n_components_; k++)
    {
      // covariance cholesky for this component
      VectorC cov_chol_vector = covariances_cholesky_.row(k);
      MatrixDD L = Eigen::Map<MatrixDD>(cov_chol_vector.data(), D, D);
      VectorD mean = means_.row(k);

#pragma omp parallel
      {
#pragma omp for
        for (unsigned int n = 0; n < n_samples_comp[k]; n++)
        {
          VectorD z = samples.row(prev_idx + n);
          VectorD Lz = L * z;
          samples.row(prev_idx + n) = Lz + mean;
        }
      }
      prev_idx += n_samples_comp[k];
    }

    return samples;
  }

  void merge(const GMM& that)
  {
    Vector new_weights = Vector::Zero(weights_.rows() + that.weights_.rows());
    MatrixXD new_means =
        MatrixXD::Zero(means_.rows() + that.means_.rows(), means_.cols());
    MatrixXC new_covariances = MatrixXC::Zero(
        covariances_.rows() + that.covariances_.rows(), covariances_.cols());
    MatrixXC new_precisions_cholesky = MatrixXC::Zero(
        precisions_cholesky_.rows() + that.precisions_cholesky_.rows(),
        precisions_cholesky_.cols());
    MatrixXC new_covariances_cholesky = MatrixXC::Zero(
        covariances_cholesky_.rows() + that.covariances_cholesky_.rows(),
        covariances_cholesky_.cols());

    new_weights << weights_.array() * support_size_,
        that.weights_.array() * that.support_size_;
    new_weights.array() /= (support_size_ + that.support_size_);
    new_weights.array() /= new_weights.sum();

    new_means << means_, that.means_;
    new_covariances << covariances_, that.covariances_;
    new_precisions_cholesky << precisions_cholesky_, that.precisions_cholesky_;
    new_covariances_cholesky << covariances_cholesky_,
        that.covariances_cholesky_;

    weights_ = new_weights;
    means_ = new_means;
    covariances_ = new_covariances;
    precisions_cholesky_ = new_precisions_cholesky;
    covariances_cholesky_ = new_covariances_cholesky;
    support_size_ += that.support_size_;
    converged_ = true;
    n_components_ += that.n_components_;
  }

  void mvnPdf(const Matrix& X, const Matrix& mu, const Matrix& sigma,
              const Matrix& Xminusmean, int k, Matrix& probs, Matrix& Linv,
              Matrix& y)
  {
    // compute determinant of covariance matrix.
    T cov_det = static_cast<T>(sigma.determinant());

    // handle the case when the determinant is too low.
    // if (cov_det < reg_covar_ * reg_covar_)
    // {
    //   cov_det = reg_covar_ * reg_covar_;
    // }

    // compute the norm factor
    T norm_factor = MVN_NORM_3 * static_cast<T>(1.0 / std::sqrt(cov_det));

    // compute the term inside the exponential
    // avoid the matrix inverse using a linear solver via Cholesky decomposition
    // solving L y = (X - µ)^T gives the solution y = L^{-1} (X - µ)^T
    // then, the term (X - µ)^T Σ^{-1} (X - µ) becomes y^T y (easy to derive)
    Matrix L = sigma.llt().matrixL();
    Linv = L.inverse();
    y = Linv * Xminusmean;
    Matrix temp = (y.array().square().colwise().sum()).array() * (-0.5);

    probs.col(k) = ((temp.array().exp()) * norm_factor).transpose();
  }

  std::tuple<Matrix, Matrix, Matrix> colorConditional(const Matrix& X)
  {
    int N = X.rows();

    Matrix ws = Matrix::Zero(N, n_components_);
    Matrix ms = Matrix::Zero(N, n_components_);
    Vector vars = Vector::Zero(n_components_);

    Matrix dev = Matrix::Zero(N, D - 1);

    // parts of the mean vector
    Matrix mu_kX = Matrix::Zero(D - 1, 1);
    T mu_kli;

    // parts of the covariance matrix
    MatrixDD sigma = Matrix::Zero(D, D);
    Matrix sigma_kXX = Matrix::Zero(D - 1, D - 1);
    Matrix sigma_kXX_inv = Matrix::Zero(D - 1, D - 1);
    Matrix sigma_kXli = Matrix::Zero(D - 1, 1);
    Matrix sigma_kliX = Matrix::Zero(1, D - 1);
    T sigma_klili;

    Matrix sigma_kliX_kXX_inv = Matrix::Zero(1, D - 1);

    // temp variable to reduce redundant computation later
    // Linv is the inverse of (lower triangular) Cholesky decomposition of
    // sigma_kXX
    Matrix Linv = Matrix::Zero(D - 1, D - 1);
    // y = L^{-1} (X - µ)^T
    Matrix y = Matrix::Zero(N, n_components_);

    for (int k = 0; k < n_components_; k++)
    {
      // parts of the mean vector
      mu_kX = means_(k, { 0, 1, 2 });
      mu_kli = static_cast<T>(means_(k, 3));

      // parts of the covariance matrix
      sigma = Eigen::Map<MatrixDD>(covariances_.row(k).data(), D, D);
      sigma_kXX = sigma({ 0, 1, 2 }, { 0, 1, 2 });
      sigma_kXli = sigma({ 0, 1, 2 }, { 3 });
      sigma_kliX = sigma({ 3 }, { 0, 1, 2 });
      sigma_klili = static_cast<T>(sigma(3, 3));

      dev = (X.rowwise() - mu_kX(0, Eigen::all)).transpose();

      mvnPdf(X, mu_kX, sigma_kXX, dev, k, ws, Linv, y);
      ws.col(k) = weights_(k) * ws.col(k);

      sigma_kXX_inv = (Linv.transpose()) * Linv;
      sigma_kliX_kXX_inv = sigma_kliX * sigma_kXX_inv;

      ms.col(k) = ((sigma_kliX_kXX_inv * dev).array() + mu_kli).transpose();
      vars(k) = (sigma_klili - (sigma_kliX_kXX_inv * sigma_kXli).value());
    }

    // clamp ws to zero
    ws = (ws.array() < reg_covar_).select(0.0, ws);

    Vector ws_sums = Vector::Zero(N);
    ws_sums << ws.rowwise().sum();

    // normalize in place
    ws.array().colwise() /= ws_sums.array();

    // nan to zeros
    ws = (ws.array().isFinite()).select(ws, static_cast<T>(0.0));

    // expected values
    Vector expected_values = Vector::Zero(N);
    expected_values = (ws.array() * ms.array()).rowwise().sum();

    // uncertainty
    Vector uncerts = Vector::Zero(N);
    uncerts = (ws.array() * (ms.array().square().rowwise() + vars.transpose().array())).rowwise().sum();
    uncerts = uncerts.array() - expected_values.array().square();

    return std::make_tuple(ws, expected_values, uncerts);
  }

  Tensor getLogResp() const
  {
    return Log_Resp__;
  }

  // Public GPU members
  Tensor Weights_;
  Tensor Means_;
  Tensor Covariances_;
  Tensor Covs_Chol_;
  Tensor Precs_Chol_;

  // CPU members
  Vector weights_;
  MatrixXD means_;
  MatrixXC covariances_;
  MatrixXC precisions_cholesky_;
  MatrixXC covariances_cholesky_;
  unsigned int support_size_;
  bool converged_ = false;
  unsigned int n_components_;
  Device device_;
  Dtype dtype_;
  T tol_;
  T reg_covar_;
  unsigned int max_iter_;
  unsigned int n_samples_ = 0;
  T likelihood_;

  // for Box-Muller sampling
  gsl_rng* r_global_;
  std::default_random_engine generator_;
  std::normal_distribution<T> normal_dist_;

  TimeProfiler tp_;

private:
  Tensor amax__;
  Tensor Log_Det_Cholesky__;
  Tensor Log_Det_Cholesky_Tmp__;
  Tensor eDiff__;
  Tensor Log_Prob_Norm__;

  Tensor Nk__;
  Tensor mDiff__;
  Tensor third_eye__;
  Tensor A__;
  Tensor B__;
  Tensor Log_Resp__;
};
