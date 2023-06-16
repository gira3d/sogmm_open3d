#include <fstream>
#include <iostream>
#include <unistd.h>

#include <self_organizing_gmm/KInit.h>
#include <sogmm_open3d/GMM.h>

using GMMf4 = GMM<float, 4>;
using KInitf4 = KInit<float, 4>;

template <typename M>
M load_csv_vector(const std::string& path)
{
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<float> values;
  uint rows = 0;
  while (std::getline(indata, line))
  {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ','))
    {
      values.push_back(std::stof(cell));
    }
    ++rows;
  }
  return Eigen::Map<const Eigen::Matrix<
      typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime>>(
      values.data(), rows, values.size() / rows);
}

template <typename M>
M load_csv_matrix(const std::string& path)
{
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<float> values;
  uint rows = 0;
  while (std::getline(indata, line))
  {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ','))
    {
      values.push_back(std::stof(cell));
    }
    ++rows;
  }
  return Eigen::Map<
      const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, Eigen::RowMajor>>(
      values.data(), rows, values.size() / rows);
}

bool test_gpu_fit_single(unsigned int K)
{
  GMMf4::MatrixXD X =
      load_csv_matrix<GMMf4::MatrixXD>("../../test_data/pcld_npy.csv");

  GMMf4::Matrix resp = load_csv_matrix<GMMf4::Matrix>(
      "../../test_data/resp_ref_" + std::to_string(K) + ".csv");

  GMMf4 gmm(K, X.rows(), "CUDA:0", false);

  GMMf4::Tensor Xt = EigenMatrixToTensor(X, gmm.device_);
  GMMf4::Tensor Respt = EigenMatrixToTensor(resp, gmm.device_);
  bool success = gmm.fit(Xt, Respt);

  if (!success)
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool test_gpu_fit_multiple()
{
  std::vector<unsigned int> K = { 300, 200, 100, 50 };

  for (int i = 0; i < K.size(); i++)
  {
    bool success = test_gpu_fit_single(K[i]);

    if (!success)
    {
      return false;
    }
  }

  return true;
}

int main(int argc, char* argv[])
{
  bool success = test_gpu_fit_multiple();

  if (success)
  {
    return EXIT_SUCCESS;
  }
  else
  {
    return EXIT_FAILURE;
  }
}