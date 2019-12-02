#include "integral_gauss.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <string>

// format of the gradient:
// dv1/dx1  dv2/dx1
// dv1/dx2  dv2/dx2

// format of dgrad:
// d2 v/dx1 dx1              d2 v/ dx1 dx2
// d2 v/ dx1 dx2             d2 v/ dx2 dx2

class NU_BASE {
public:
  virtual Eigen::Vector2d operator()(const Eigen::Vector2d &X) const = 0;
  virtual Eigen::MatrixXd grad(const Eigen::Vector2d &X) const = 0;
  virtual double div(const Eigen::Vector2d &X) const = 0;
  virtual Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const = 0;
  virtual Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const = 0;
};

class NU_XYMN_1 : public NU_BASE {
  const int m;
  const int n;

public:
  NU_XYMN_1(int m_, int n_) : m(m_), n(n_){};

  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    return Eigen::Vector2d(1+sin(m * x) * sin(n * y), 0);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << m * cos(m * x) * sin(n * y), 0, n * cos(n * y) * sin(m * x), 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return m * cos(m * x) * sin(n * y);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << -m * m * sin(m * x) * sin(n * y), m * n * cos(m * x) * cos(n * y),
        m * n * cos(m * x) * cos(n * y), -n * n * sin(m * x) * sin(n * y);
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_XYMN_2 : public NU_BASE {
  const int m;
  const int n;

public:
  NU_XYMN_2(int m_, int n_) : m(m_), n(n_){};

  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    return Eigen::Vector2d(0, 1+sin(m * x) * sin(n * y));
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 0, m * cos(m * x) * sin(n * y), 0, n * cos(n * y) * sin(m * x);
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return n * sin(m * x) * cos(n * y);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << -m * m * sin(m * x) * sin(n * y), m * n * cos(m * x) * cos(n * y),
        m * n * cos(m * x) * cos(n * y), -n * n * sin(m * x) * sin(n * y);
    return M;
  }
};

class NU_POLY_1 : public NU_BASE {
  const int m;
  const int n;
public:
  NU_POLY_1(int m_,int n_):m(m_),n(n_){};

  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    return Eigen::Vector2d(std::pow(x,m)*std::pow(y,n), 0);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    double powxm1, powym1;
    if (m==0)
      powxm1 = 0;
    else
      powxm1 = m * std::pow(x,m-1);

    if (n==0)
      powym1 = 0;
    else
      powym1 = n * std::pow(y,n-1);
    M << powxm1*std::pow(y,n), 0, std::pow(x,m)*powym1, 0;

    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (m==0)
      return 0;
    else
      return m*std::pow(x,m-1)*std::pow(y,n);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    double M11,M12,M22;

    if (m==0 or m==1)
      M11 = 0;
    else
      M11 = m*(m-1)*std::pow(x,m-2)*std::pow(y,n);

    if (m==0 or n==0)
      M12 = 0;
    else
      M12 = m*n*std::pow(x,m-1)*std::pow(y,n-1);

    if (n==0 or n==1)
      M22 = 0;
    else
      M22 = n*(n-1)*std::pow(x,m)*std::pow(y,n-2);

    M << M11, M12, M12, M22;
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_POLY_2 : public NU_BASE {
  const int m;
  const int n;
public:
  NU_POLY_2(int m_,int n_):m(m_),n(n_){};

  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    return Eigen::Vector2d(0, std::pow(x,m)*std::pow(y,n));
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    double powxm1, powym1;
    if (m==0)
      powxm1 = 0;
    else
      powxm1 = m * std::pow(x,m-1);

    if (n==0)
      powym1 = 0;
    else
      powym1 = n * std::pow(y,n-1);
    M << 0, powxm1*std::pow(y,n), 0, std::pow(x,m)*powym1;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (n==0)
      return 0;
    else
      return n *std::pow(x,m)*std::pow(y,n-1);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    double M11,M12,M22;

    if (m==0 or m==1)
      M11 = 0;
    else
      M11 = m*(m-1)*std::pow(x,m-2)*std::pow(y,n);

    if (m==0 or n==0)
      M12 = 0;
    else
      M12 = m*n*std::pow(x,m-1)*std::pow(y,n-1);

    if (n==0 or n==1)
      M22 = 0;
    else
      M22 = n*(n-1)*std::pow(x,m)*std::pow(y,n-2);

    M << M11, M12, M12, M22;
    return M;
  }
};

std::pair<double,double> H1Dnorm(const std::shared_ptr<NU_BASE> &nu1,
               const std::shared_ptr<NU_BASE> &nu2) {
  double ll = -5 * M_PI;
  double ul = 5 * M_PI;
  auto LL = [&](double x) { return -5 * M_PI; };
  auto UL = [&](double x) { return 5 * M_PI; };
  // L2 norm first
  auto l2integrand = [&](double x, double y) {
    Eigen::Vector2d pt(x, y);
    return nu1->operator()(pt).dot(nu2->operator()(pt));
  };

  // H1 seminorm
  auto h1semiintegrand = [&](double x, double y) {
    Eigen::Vector2d pt(x, y);
    return nu1->grad(pt).cwiseProduct(nu2->grad(pt)).sum();
  };
  unsigned order = 10;
  double l2norm =
      parametricbem2d::ComputeDoubleIntegral(l2integrand, ll, ul, LL, UL, order);
  double h1seminorm = parametricbem2d::ComputeDoubleIntegral(
      h1semiintegrand, ll, ul, LL, UL, order);

  return std::make_pair(l2norm , h1seminorm);
}

Eigen::MatrixXd
GramianMatrix(const std::vector<std::shared_ptr<NU_BASE>> &basis) {
  //std::cout << "gramian matrix function called" << std::endl;
  unsigned N = basis.size();
  Eigen::MatrixXd G = Eigen::MatrixXd::Constant(N, N, 0);
  Eigen::MatrixXd L2 = Eigen::MatrixXd::Constant(N,N,0);
  Eigen::MatrixXd H1 = Eigen::MatrixXd::Constant(N,N,0);
  double l2,h1;
  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      //std::cout << "computing norm" << std::endl;
      std::tie(l2,h1) = H1Dnorm(basis[i], basis[j]);
      //std::cout << "computed norm! = " << norm << std::endl;
      L2(i,j) += l2;
      H1(i,j) += h1;
      G(i, j) += l2+h1;
    }
  }
  std::cout << "L2 mat \n" << L2 << std::endl;
  std::cout << "H1 mat \n" << H1 << std::endl;
  return G;
}

Eigen::MatrixXd GramianMatrixSimple(unsigned M, unsigned halfside) {
  //unsigned halfside = 5; // halfside for the square divided by pi!
  Eigen::MatrixXd gmat = Eigen::MatrixXd::Constant(2*M*M,2*M*M,0);
  Eigen::VectorXd l2diag = Eigen::VectorXd::Constant(M*M,halfside*halfside*M_PI*M_PI);
  Eigen::MatrixXd small = Eigen::MatrixXd::Constant(M*M,M*M,0);
  small.diagonal() = l2diag;
  Eigen::VectorXd h1diag = Eigen::VectorXd::Constant(M*M,0);
  for (unsigned i = 0 ; i < M ; ++i) {
    for (unsigned j = 0 ; j < M ; ++j) {
      h1diag(i*M+j) = halfside*halfside*M_PI*M_PI*((i+1)*(i+1)+(j+1)*(j+1));
    }
  }
  std::cout << "l2diag: " << l2diag << std::endl;
  std::cout << "h2diag: " << h1diag << std::endl;
  small.diagonal()+=h1diag;
  gmat.block(0,0,M*M,M*M) = small;
  gmat.block(M*M,M*M,M*M,M*M) = small;
  return gmat;
}

int main() {
  unsigned M = 5;
  std::string fname = "gramat.txt";
  std::ofstream out(fname);
  std::vector<std::shared_ptr<NU_BASE>> basis(2 * M * M);
  std::vector<std::shared_ptr<int>> test(2 * M * M);
  // std::shared_ptr<NU_BASE> test(new NU_XYMN_1(1,1));
  // Eigen::Vector2d velocity =
  // test->operator()(Eigen::Vector2d(M_PI/2,M_PI/2)); std::cout << "Test: " <<
  // velocity << std::endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) {
      // basis[j + i * M]=(new NU_XYMN_1(i + 1, j + 1));
      basis[M * M + j + i * M] = std::make_shared<NU_POLY_2> (i,j);
          //std::shared_ptr<NU_XYMN_2>(new NU_XYMN_2(i + 1, j + 1));

      basis[j + i * M] = std::make_shared<NU_POLY_1> (i, j);
          //std::shared_ptr<NU_XYMN_1>(new NU_XYMN_1(i + 1, j + 1));
      // test[j+i*M] = std::shared_ptr<int> (new int(10));
    }
  }
  // for (unsigned i = 0 ; i < 2*M*M ; ++i) {
  // std::cout << "yoyoma: " << *test[i] << std::endl;
  //}
  //std::cout << "basis stored" << std::endl;
  Eigen::MatrixXd gramat = GramianMatrix(basis);
  //gramat = Eigen::MatrixXd::Constant(2*M*M,2*M*M,1.5);
  //std::cout << "gramian computed" << std::endl;
  std::cout << "Gramian Matrix: \n" << gramat << std::endl;
  out << gramat << std::endl;
  //std::cout << "simple version \n" << GramianMatrixSimple(M,5) << std::endl;
  return 0;
}
