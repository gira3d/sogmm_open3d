#pragma once

#define BUILD_CUDA_MODULE

#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Dense>

#include <sogmm_open3d/SOGMMGPU.h>

#define LOG_2_M_PI 1.83788

namespace o3d = open3d;

namespace sogmm
{
  namespace gpu
  {
    template <typename T, uint32_t D>
    class EM
    {
    public:
      static constexpr uint32_t C = D * D;

      using Ptr = std::shared_ptr<EM<T, D>>;
      using ConstPtr = std::shared_ptr<const EM<T, D>>;

      using Container = SOGMM<T, D>;

      using Tensor = typename Container::Tensor;
      using SizeVector = typename Container::SizeVector;
      using Device = typename Container::Device;
      using Dtype = typename Container::Dtype;

      EM()
      {
        tol_ = 1e-3;
        reg_covar_ = 1e-6;
        max_iter_ = 100;

        device_ = Device("CUDA:0");
        dtype_ = Dtype::template FromType<T>();
      }

      ~EM()
      {
      }

      void initialize(const unsigned int &n_samples, const unsigned int &n_components)
      {
        unsigned int N = n_samples;
        unsigned int K = n_components;

        amax__ = Tensor::Zeros({N, 1, 1}, dtype_, device_);
        Log_Det_Cholesky__ = Tensor::Zeros({K, 1}, dtype_, device_);
        Log_Det_Cholesky_Tmp__ = Tensor::Zeros({K, D}, dtype_, device_);
        eDiff__ = Tensor::Zeros({N, K, D, 1}, dtype_, device_);
        Log_Prob_Norm__ = Tensor::Zeros({N, 1, 1}, dtype_, device_);

        Nk__ = Tensor::Zeros({1, K}, dtype_, device_);
        mDiff__ = Tensor::Zeros({K, D, N}, dtype_, device_);
        Log_Resp__ = Tensor::Zeros({N, K, 1}, dtype_, device_);
      }

      void logSumExp(const Tensor &in, const unsigned int dim, Tensor &out)
      {
        amax__ = in.Max({dim}, true);
        ((in - amax__).Exp()).Sum_({dim}, true, out);
        out.Log_();
        out.Add_(amax__);
      }

      void estimateWeightedLogProb(const Tensor &Xt, const Container &sogmm,
                                   Tensor &Weighted_Log_Prob)
      {
        // Term 2 of Equation (3.14)
        // Log_Det_Cholesky__ = ((GetDiagonal(Precs_Chol_[0]).Log()).Sum({ 1 }, true))
        //                         .Reshape({ 1, n_components_, 1 });

        SizeVector Xt_shape = Xt.GetShape();
        unsigned int N = Xt_shape[0];
        unsigned int K = sogmm.n_components_;

        Log_Det_Cholesky__.template Fill<float>(0.0);
        Tensor Log_Det_Cholesky_View = (GetDiagonal(sogmm.precisions_cholesky_[0]));
        Log_Det_Cholesky_Tmp__.CopyFrom(Log_Det_Cholesky_View);
        Log_Det_Cholesky_Tmp__.Log_();
        Log_Det_Cholesky_Tmp__.Sum_({1}, true, Log_Det_Cholesky__);

        // Diff, PDiff, Log_Gaussian_Prob are all terms for the first
        // term in Equation (3.14)
        eDiff__ = (Xt.Reshape({N, 1, D}) - sogmm.means_).Reshape({N, K, D, 1});

        eDiff__ = (sogmm.precisions_cholesky_.Mul(eDiff__)).Sum({2}, true).Reshape({N, K, D, 1});

        // Equation (3.14), output of estimateLogGaussianProb
        (eDiff__.Mul(eDiff__)).Sum_({2}, false, Weighted_Log_Prob);
        Weighted_Log_Prob.Add_(D * LOG_2_M_PI);
        Weighted_Log_Prob.Mul_(-0.5);
        Weighted_Log_Prob.Add_(Log_Det_Cholesky__.Reshape({1, K, 1}));

        // This is the first two terms in Equation (3.7)
        Weighted_Log_Prob.Add_(sogmm.weights_.Log());
      }

      void eStep(const Tensor &Xt, const Container &sogmm)
      {
        unsigned int K = sogmm.n_components_;

        estimateWeightedLogProb(Xt, sogmm, Log_Resp__);

        Log_Prob_Norm__.template Fill<float>(0.0);
        logSumExp(Log_Resp__, 1, Log_Prob_Norm__);

        Log_Resp__.Sub_(Log_Prob_Norm__);

        likelihood_ = Log_Prob_Norm__.Mean({0, 1, 2}, false).template Item<T>();
      }

      void mStep(const Tensor &Xt, const Tensor &Respt, Container &sogmm)
      {
        SizeVector Xt_shape = Xt.GetShape();
        unsigned int N = Xt_shape[0];
        unsigned int K = sogmm.n_components_;

        // initialize tensor for weights
        Nk__.template Fill<float>(0.0);
        Respt.Sum_({0}, true, Nk__);
        sogmm.weights_ = Nk__.T();
        sogmm.weights_.Add_(Tensor::Ones({K, 1}, dtype_, device_) * 10 *
                            std::numeric_limits<T>::epsilon());

        // update means
        sogmm.means_ = (Respt.T().Matmul(Xt)).Div(sogmm.weights_).Reshape(sogmm.means_.GetShape());

        // update covariances
        mDiff__ = (Xt.Reshape({N, 1, D}) - sogmm.means_).AsStrided({K, D, N}, {D, 1, D * K});
        sogmm.covariances_ = mDiff__.MatmulBatched(mDiff__.Transpose(1, 2) *
                                                   Respt.AsStrided({K, N, 1}, {1, K, 1}))
                                 .Reshape(sogmm.covariances_.GetShape());
        sogmm.covariances_.Div_(sogmm.weights_.Reshape({1, K, 1, 1}));

        // add reg_covar_ along the diagonal for Covariances_
        sogmm.covariances_[0].AsStrided({K, D}, {C, D + 1}).Add_(reg_covar_);

        // update weights
        sogmm.weights_.Div_(N);
        sogmm.weights_.Div_(sogmm.weights_.Sum({0}, false));

        // update precision and covariance cholesky
        sogmm.updateCholesky();
      }

      bool fit(const Tensor &Xt, const Tensor &Respt, Container &sogmm)
      {
        unsigned int K = sogmm.n_components_;

        SizeVector Xt_shape = Xt.GetShape();
        unsigned int N = Xt_shape[0];

        if (N <= 1)
        {
          throw std::runtime_error("fit: number of samples should be greater than 1.");
        }

        if (K <= 0)
        {
          throw std::runtime_error("fit: number of components should be greater than 0.");
        }

        if (N < K)
        {
          throw std::runtime_error("fit: number of components is " +
                                   std::to_string(K) +
                                   ". It should be strictly smaller than the "
                                   "number of points: " +
                                   std::to_string(N));
        }

        initialize(N, K);

        mStep(Xt, Respt, sogmm);

        T lower_bound = -std::numeric_limits<T>::infinity();
        for (unsigned int n_iter = 0; n_iter <= max_iter_; n_iter++)
        {
          T prev_lower_bound = lower_bound;

          // E step
          eStep(Xt, sogmm);

          // M step
          mStep(Xt, Log_Resp__.Exp().Reshape({N, K}), sogmm);

          // convergence check
          lower_bound = likelihood_;
          T change = lower_bound - prev_lower_bound;
          if (!std::isinf(change) && std::abs(change) < tol_)
          {
            converged_ = true;
            break;
          }
        }

        // o3d::core::MemoryManagerCached::ReleaseCache(device_);

        if (converged_)
        {
          return true;
        }
        else
        {
          return false;
        }
      }

      static void scoreSamples(const Tensor &Xt, const Container &sogmm,
                               Tensor &Per_Sample_Log_Likelihood)
      {
        SizeVector Xt_shape = Xt.GetShape();
        unsigned int N = Xt_shape[0];
        unsigned int K = sogmm.n_components_;

        Tensor Weighted_Log_Prob = Tensor::Zeros({N, K, 1}, sogmm.dtype_, sogmm.device_);
        Tensor eDiff = Tensor::Zeros({N, K, D, 1}, sogmm.dtype_, sogmm.device_);
        Tensor Log_Det_Cholesky = Tensor::Zeros({K, 1}, sogmm.dtype_, sogmm.device_);
        Tensor Log_Det_Cholesky_Tmp = Tensor::Zeros({K, D}, sogmm.dtype_, sogmm.device_);

        Tensor Log_Det_Cholesky_View = (GetDiagonal(sogmm.precisions_cholesky_[0]));

        Log_Det_Cholesky.template Fill<float>(0.0);
        Log_Det_Cholesky_Tmp.CopyFrom(Log_Det_Cholesky_View);
        Log_Det_Cholesky_Tmp.Log_();
        Log_Det_Cholesky_Tmp.Sum_({1}, true, Log_Det_Cholesky);

        eDiff = (Xt.Reshape({N, 1, D}) - sogmm.means_).Reshape({N, K, D, 1});
        eDiff = (sogmm.precisions_cholesky_.Mul(eDiff)).Sum({2}, true).Reshape({N, K, D, 1});

        (eDiff.Mul(eDiff)).Sum_({2}, false, Weighted_Log_Prob);
        Weighted_Log_Prob.Add_(D * LOG_2_M_PI);
        Weighted_Log_Prob.Mul_(-0.5);
        Weighted_Log_Prob.Add_(Log_Det_Cholesky.Reshape({1, K, 1}));

        Weighted_Log_Prob.Add_(sogmm.weights_.Log());

        Tensor amax = Tensor::Zeros({N, 1, 1}, sogmm.dtype_, sogmm.device_);

        amax = Weighted_Log_Prob.Max({1}, true);
        ((Weighted_Log_Prob - amax).Exp()).Sum_({1}, true, Per_Sample_Log_Likelihood);
        Per_Sample_Log_Likelihood.Log_();
        Per_Sample_Log_Likelihood.Add_(amax);
      }

      bool converged_ = false;

      Device device_;
      Dtype dtype_;

      T tol_;
      T reg_covar_;
      unsigned int max_iter_;
      T likelihood_;

    private:
      Tensor amax__;
      Tensor Log_Det_Cholesky__;
      Tensor Log_Det_Cholesky_Tmp__;
      Tensor eDiff__;
      Tensor Log_Prob_Norm__;

      Tensor Nk__;
      Tensor mDiff__;
      Tensor Log_Resp__;
    };
  }
}
