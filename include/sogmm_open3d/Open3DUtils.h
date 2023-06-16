#pragma once

#include <open3d/core/Device.h>
#include <open3d/core/CUDAUtils.h>

namespace o3d = open3d;
using Tensor = o3d::core::Tensor;
using Device = o3d::core::Device;
using Dtype = o3d::core::Dtype;
using SizeVector = o3d::core::SizeVector;

template <class Derived>
Tensor EigenMatrixToTensor(const Eigen::MatrixBase<Derived>& matrix,
                           const Device& device)
{
  typedef typename Derived::Scalar Scalar;
  Dtype dtype = Dtype::FromType<Scalar>();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      matrix_row_major = matrix;
  return Tensor(matrix_row_major.data(), { matrix.rows(), matrix.cols() },
                dtype, device);
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
TensorToEigenMatrix(const Tensor& tensor)
{
  static_assert(std::is_same<T, double>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, int>::value,
                "Only supports double, float and int (MatrixXd, MatrixXf and "
                "MatrixXi).");
  Dtype dtype = Dtype::FromType<T>();

  SizeVector dim = tensor.GetShape();
  if (dim.size() != 2)
  {
    o3d::utility::LogError(
        " [TensorToEigenMatrix]: Number of dimensions supported = 2, "
        "but got {}.",
        dim.size());
  }

  Tensor tensor_cpu_contiguous = tensor.Contiguous().To(Device("CPU:0"), dtype);
  T* data_ptr = tensor_cpu_contiguous.GetDataPtr<T>();

  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      eigen_matrix(data_ptr, dim[0], dim[1]);

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eigen_matrix_copy(eigen_matrix);
  return eigen_matrix_copy;
}

Tensor GetDiagonal(const Tensor& A)
{
  std::vector<int64_t> shape = A.GetShape();

  if (shape.size() == 3)
  {
    if (shape[1] != shape[2])
    {
      o3d::utility::LogError(
          "GetDiagonal expects a K x D x D Tensor with <= 3 dimensions, but "
          "the "
          "Tensor has K x D x J dimensions.");
    }
    int K = shape[0];
    int D = shape[1];
    int DD = D * D;
    return A.AsStrided({ K, D }, { DD, D + 1 });
  }
  else if (shape.size() == 2)
  {
    std::vector<int64_t> strides = A.GetStrides();
    int K = shape[0];
    return A.AsStrided({ K }, { strides[0] + strides[1] });
  }
  else
  {
    o3d::utility::LogError("GetDiagonal not implemented 1D");
  }
}