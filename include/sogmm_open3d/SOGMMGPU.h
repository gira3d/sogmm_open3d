#pragma once

#define BUILD_CUDA_MODULE

#include <self_organizing_gmm/SOGMMCPU.h>

#include <open3d/Open3D.h>
#include <open3d/core/CUDAUtils.h>
#include <sogmm_open3d/Open3DUtils.h>

namespace o3d = open3d;

namespace sogmm
{
  namespace gpu
  {
    /// @brief Container for SOGMM on the GPU using Open3D tensors.
    /// @tparam T Datatype (e.g., float, double)
    /// @tparam D Number of dimensions (e.g., 1, 2, 3, 4)
    template <typename T, uint32_t D>
    struct SOGMM
    {
      using Ptr = std::shared_ptr<SOGMM<T, D>>;
      using ConstPtr = std::shared_ptr<const SOGMM<T, D>>;

      static constexpr uint32_t C = D * D;

      using HostContainer = sogmm::cpu::SOGMM<T, D>;

      using Tensor = o3d::core::Tensor;
      using SizeVector = o3d::core::SizeVector;
      using Device = o3d::core::Device;
      using Dtype = o3d::core::Dtype;

      Tensor weights_;
      Tensor means_;
      Tensor covariances_;
      Tensor covariances_cholesky_;
      Tensor precisions_cholesky_;

      uint32_t support_size_;
      uint32_t n_components_;

      Device device_;
      Dtype dtype_;

      /// @brief Default constructor.
      /// @details Initially there are no points in the support and no components.
      SOGMM()
      {
        support_size_ = 0;
        n_components_ = 0;

        device_ = o3d::core::Device("CUDA:0");
        dtype_ = Dtype::FromType<T>();
      }

      /// @brief Copy constructor.
      /// @param that SOGMM to copy from.
      SOGMM(const SOGMM &that)
      {
        this->device_ = that.device_;
        this->dtype_ = that.dtype_;

        this->support_size_ = that.support_size_;
        this->n_components_ = that.n_components_;

        this->weights_ = that.weights_;
        this->normalizeWeights();

        this->means_ = that.means_;
        this->covariances_ = that.covariances_;
        this->covariances_cholesky_ = that.covariances_cholesky_;
        this->precisions_cholesky_ = that.precisions_cholesky_;
      }

      /// @brief Initialization with known number of components.
      /// @param n_components Number of components in the SOGMM.
      SOGMM(const uint32_t &n_components)
      {
        device_ = o3d::core::Device("CUDA:0");
        dtype_ = Dtype::FromType<T>();

        if (n_components <= 0)
        {
          throw std::runtime_error("Number of components should be atleast 1.");
        }

        n_components_ = n_components;

        weights_ = Tensor::Zeros({1, n_components, 1}, dtype_, device_);
        means_ = Tensor::Zeros({1, n_components, D}, dtype_, device_);
        covariances_ = Tensor::Zeros({1, n_components, D, D}, dtype_, device_);
        covariances_cholesky_ = Tensor::Zeros({1, n_components, D, D}, dtype_, device_);
        precisions_cholesky_ = Tensor::Zeros({1, n_components, D, D}, dtype_, device_);
      }

      /// @brief Initialization with known SOGMM parameters.
      /// @param weights Weights of the SOGMM. Should be normalized to 1.0.
      /// @param means Means of the SOGMM.
      /// @param covariances Covariances of the SOGMM.
      /// @param support_size Number of points in the support of the SOGMM.
      SOGMM(const Tensor &weights, const Tensor &means,
            const Tensor &covariances, const uint32_t &support_size)
      {
        device_ = o3d::core::Device("CUDA:0");
        dtype_ = Dtype::FromType<T>();

        if (support_size <= 1)
        {
          throw std::runtime_error("The support size for this SOGMM is less than or equal to 1.");
        }

        support_size_ = support_size;
        n_components_ = means.GetShape()[0];

        weights_ = Tensor::Zeros({1, n_components_, 1}, dtype_, device_);
        means_ = Tensor::Zeros({1, n_components_, D}, dtype_, device_);
        covariances_ = Tensor::Zeros({1, n_components_, D, D}, dtype_, device_);
        covariances_cholesky_ = Tensor::Zeros({1, n_components_, D, D}, dtype_, device_);
        precisions_cholesky_ = Tensor::Zeros({1, n_components_, D, D}, dtype_, device_);

        weights_ = weights;
        means_ = means;
        covariances_ = covariances;

        updateCholesky(covariances_);
      }

      void updateCholesky(const Tensor &covariances)
      {
        covariances_cholesky_ = covariances[0].LLTBatched().Reshape(covariances_cholesky_.GetShape());

        Tensor I = Tensor::Zeros({n_components_, D, D}, dtype_, device_);
        I.AsStrided({n_components_, D}, {C, D + 1}).Fill(static_cast<T>(1));

        Tensor A = covariances_cholesky_[0].Transpose(1, 2).MatmulBatched(covariances_cholesky_[0]);
        Tensor B = covariances_cholesky_[0].Transpose(1, 2).MatmulBatched(I);

        precisions_cholesky_ = A.SolveLLTBatched(B).Reshape(precisions_cholesky_.GetShape());
      }

      void updateCholesky()
      {
        updateCholesky(covariances_);
      }

      /// @brief Normalize the weights of this SOGMM.
      void normalizeWeights()
      {
        weights_.Div_(weights_.Sum({0}, false));
      }

      /// @brief Normalize weights in place
      /// @param weights Weights tensor to be modified
      void normalizeWeights(Tensor &weights) const
      {
        weights.Div_(weights.Sum({0}, false));
      }

      /// @brief Return a container on the host for this SOGMM.
      /// @param sogmm HostContainer (SOGMM<T, D>)
      void toHost(HostContainer &sogmm) const
      {
        sogmm.support_size_ = support_size_;
        sogmm.n_components_ = n_components_;

        sogmm.weights_ = TensorToEigenMatrix<T>(weights_.Reshape({n_components_, 1}));
        sogmm.means_ = TensorToEigenMatrix<T>(means_.Reshape({n_components_, D}));
        sogmm.covariances_ =
            TensorToEigenMatrix<T>(covariances_.Reshape({n_components_, C}));
        sogmm.covariances_cholesky_ =
            TensorToEigenMatrix<T>(covariances_cholesky_.Reshape({n_components_, C}));
        sogmm.precisions_cholesky_ =
            TensorToEigenMatrix<T>(precisions_cholesky_.Reshape({n_components_, C}));
      }

      /// @brief Fill the SOGMM data from host.
      /// @param from HostContainer (SOGMM<T, D>) containing SOGMM on CPU.
      void fromHost(const HostContainer &from)
      {
        device_ = o3d::core::Device("CUDA:0");
        dtype_ = Dtype::FromType<T>();

        n_components_ = from.n_components_;
        support_size_ = from.support_size_;

        weights_ = Tensor::Zeros({1, n_components_, 1}, dtype_, device_);
        means_ = Tensor::Zeros({1, n_components_, D}, dtype_, device_);
        covariances_ = Tensor::Zeros({1, n_components_, D, D}, dtype_, device_);
        covariances_cholesky_ = Tensor::Zeros({1, n_components_, D, D}, dtype_, device_);
        precisions_cholesky_ = Tensor::Zeros({1, n_components_, D, D}, dtype_, device_);

        weights_ = EigenMatrixToTensor(from.weights_, device_)
                       .Reshape(weights_.GetShape());
        means_ = EigenMatrixToTensor(from.means_, device_)
                     .Reshape(means_.GetShape());
        covariances_ = EigenMatrixToTensor(from.covariances_, device_)
                           .Reshape(covariances_.GetShape());
        covariances_cholesky_ = EigenMatrixToTensor(from.covariances_cholesky_, device_)
                                    .Reshape(covariances_.GetShape());
        precisions_cholesky_ = EigenMatrixToTensor(from.precisions_cholesky_, device_)
                                   .Reshape(covariances_.GetShape());
      }

      /// @brief Merge the input GMM into this GMM.
      /// @param that Input GMM that needs to be merged.
      void merge(const SOGMM<T, D> &that)
      {
        Tensor S1 = weights_.Mul(support_size_).Reshape({1, n_components_, 1});
        Tensor S2 = that.weights_.Mul(that.support_size_).Reshape({1, that.n_components_, 1});

        n_components_ += that.n_components_;

        Tensor new_weights = Tensor::Zeros({1, n_components_, 1}, dtype_, device_);
        Tensor new_means = Tensor::Zeros({1, n_components_, D}, dtype_, device_);
        Tensor new_covariances = Tensor::Zeros({1, n_components_, D, D}, dtype_, device_);
        Tensor new_covariances_cholesky = Tensor::Zeros({1, n_components_, D, D}, dtype_, device_);
        Tensor new_precisions_cholesky = Tensor::Zeros({1, n_components_, D, D}, dtype_, device_);

        new_weights = S1.Append(S2, 1);
        new_weights.Div_(support_size_ + that.support_size_);
        new_weights.Div_(new_weights.Sum({1}, false));

        new_means = means_.Append(that.means_, 1);
        new_covariances = covariances_.Append(that.covariances_, 1);
        new_covariances_cholesky = covariances_cholesky_.Append(that.covariances_cholesky_, 1);
        new_precisions_cholesky = precisions_cholesky_.Append(that.precisions_cholesky_, 1);

        weights_ = new_weights;
        means_ = new_means;
        covariances_ = new_covariances;
        precisions_cholesky_ = new_precisions_cholesky;
        covariances_cholesky_ = new_covariances_cholesky;

        support_size_ += that.support_size_;
      }
    };

    template <typename T>
    SOGMM<T, 3> extractXpart(const SOGMM<T, 4> &input)
    {
      SOGMM<T, 3> sogmm3(input.n_components_);

      sogmm3.weights_ = input.weights_;

      sogmm3.means_ = input.means_.Slice(2, 0, 3);
      sogmm3.covariances_[0] = input.covariances_[0].AsStrided({input.n_components_, 3, 3}, {16, 4, 1});

      sogmm3.updateCholesky(sogmm3.covariances_);

      return sogmm3;
    }
  }
}