#pragma once

#include <memory>
#include <random>

#include <Eigen/Dense>
#include <gsl/gsl_randist.h>

#include <self_organizing_gmm/SOGMMInference.h>
#include <sogmm_open3d/EM.h>

namespace sogmm
{
  namespace gpu
  {
    template <typename T>
    class SOGMMInference
    {
    public:
      template <uint32_t D>
      using Container = SOGMM<T, D>;
      template <uint32_t D>
      using HostContainer = sogmm::cpu::SOGMM<T, D>;

      template <uint32_t D>
      using VectorD = typename HostContainer<D>::VectorD;
      template <uint32_t D>
      using VectorC = typename HostContainer<D>::VectorC;
      template <uint32_t D>
      using MatrixDD = typename HostContainer<D>::MatrixDD;
      template <uint32_t D>
      using MatrixXD = typename HostContainer<D>::MatrixXD;
      template <uint32_t D>
      using MatrixDX = typename HostContainer<D>::MatrixDX;
      template <uint32_t D>
      using MatrixXC = typename HostContainer<D>::MatrixXC;

      using Vector = typename HostContainer<4>::Vector;
      using Matrix = typename HostContainer<4>::Matrix;

      using CPUInfEng = sogmm::cpu::SOGMMInference<T>;

      template <uint32_t D>
      using EM = EM<T, D>;

      // for Box-Muller sampling
      CPUInfEng cpu_inf_;

      // reusable containers for GPU data
      Tensor Xt_;
      Tensor scores_;

      SOGMMInference()
      {
        cpu_inf_ = CPUInfEng();
      }

      MatrixXD<4> generatePointCloud4D(const Container<4> &sogmm,
                                       const unsigned int &N,
                                       double sigma)
      {
        HostContainer<4> sogmm_cpu;
        sogmm.toHost(sogmm_cpu);
        return cpu_inf_.generatePointCloud4D(sogmm_cpu, N, sigma);
      }

      MatrixXD<3> generatePointCloud3D(const Container<4> &sogmm,
                                       const unsigned int &N,
                                       double sigma)
      {
        HostContainer<4> sogmm_cpu;
        sogmm.toHost(sogmm_cpu);
        return cpu_inf_.generatePointCloud3D(sogmm_cpu, N, sigma);
      }

      MatrixXD<4> reconstruct(const Container<4> sogmm,
                              const unsigned int &N,
                              double sigma)
      {
        HostContainer<4> sogmm_cpu;
        sogmm.toHost(sogmm_cpu);

        return cpu_inf_.reconstruct(sogmm_cpu, N, sigma);
      }

      Vector score3D(const MatrixXD<3> &X, const Container<4> &sogmm)
      {
        // Extract the (x, y, z) part from SOGMM
        Container<3> sogmm3 = extractXpart(sogmm);

        unsigned int N = X.rows();

        Xt_ = EigenMatrixToTensor(X, sogmm.device_);
        scores_ = Tensor::Zeros({N, 1, 1}, sogmm.dtype_, sogmm.device_);

        EM<3>::scoreSamples(Xt_, sogmm3, scores_);

        return TensorToEigenMatrix<T>(scores_.Reshape({N, 1}));
      }

      Vector score4D(const MatrixXD<4> &X, const Container<4> &sogmm)
      {
        unsigned int N = X.rows();

        Xt_ = EigenMatrixToTensor(X, sogmm.device_);
        scores_ = Tensor::Zeros({N, 1, 1}, sogmm.dtype_, sogmm.device_);

        EM<4>::scoreSamples(Xt_, sogmm, scores_);

        return TensorToEigenMatrix<T>(scores_.Reshape({N, 1}));
      }
    };
  }
}