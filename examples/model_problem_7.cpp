#include "BoundaryMesh.hpp"
#include "force_calculation.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>

// format of the gradient:
// dv1/dx1  dv2/dx1
// dv1/dx2  dv2/dx2

// format of dgrad:
// d2 v/dx1 dx1              d2 v/ dx1 dx2
// d2 v/ dx1 dx2             d2 v/ dx2 dx2

bool inner(double x, double y) { return (fabs(x) < 2 && fabs(y) < 2); }

class G {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x*x+y*y);
    double phi = xy_to_phi(x,y);
    return std::pow(r,2./3.)*sin(2./3.*phi);
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x*x+y*y);
    double phi = xy_to_phi(x,y);
    return 2./3.*std::pow(r,-1./3.)*Eigen::Vector2d(-sin(phi/3.),cos(phi/3.));
  }
};

class GS {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x*x+y*y);
    double phi = xy_to_phi(x,y);
    return r*sin(2./3.*phi);
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x*x+y*y);
    double phi = xy_to_phi(x,y);
    return Eigen::Vector2d(sin(2.*phi/3.)*cos(phi)-2./3.*cos(2./3.*phi)*sin(phi),
                           sin(2.*phi/3.)*sin(phi)+2./3.*cos(2./3.*phi)*cos(phi));
  }
};

class GNS {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x*x+y*y);
    if (r<1) {
      return 0.;
    }
    else {
      double phi = xy_to_phi(x,y);
      return sin(2./3.*phi);
    }
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x*x+y*y);
    if (r<1) {
      return Eigen::Vector2d(0,0);
    }
    else {
      double phi = xy_to_phi(x,y);
      return Eigen::Vector2d(-2./3.*cos(2./3.*phi)*sin(phi),
                             2./3.*cos(2./3.*phi)*cos(phi));
    }
  }
};

template <int m, int n> class NU_XYMN_1 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    return Eigen::Vector2d(sin(m * x) * sin(n * y), 0);
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

template <int m, int n> class NU_XYMN_2 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    return Eigen::Vector2d(0, sin(m * x) * sin(n * y));
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

int main() {
  G g;
  //GS g;
  //GNS g;

  unsigned m = MM;
  unsigned n = NN;

  #if VEL == 1
  std::string filename("mp7xymn1_");
  filename += to_string(m)+"_"+to_string(n);

  std::ofstream out(filename);
  std::cout << "pacman, nu_xymn_1" << std::endl;
  out << "#pacman, nu_xymn_1" << std::endl;
  NU_XYMN_1<MM,NN> nu;
  #endif

  #if VEL == 2
  std::string filename("mp7xymn2_");
  filename += to_string(m)+"_"+to_string(n);

  std::ofstream out(filename);
  std::cout << "pacman, nu_xymn_2" << std::endl;
  out << "#pacman, nu_xymn_2" << std::endl;
  NU_XYMN_2<MM,NN> nu;
  #endif

  std::cout << "#MM NN: " << MM << " " << NN << std::endl;
  out << "#MM NN: " << MM << " " << NN << std::endl;

  std::cout << "#Linear Mesh!" << std::endl;
  std::cout << "#g const" << std::endl;
  out << "#Linear Mesh!" << std::endl;
  out << "#g const" << std::endl;

  // pacman
  Eigen::Vector2d B(0, -1);
  Eigen::Vector2d R(1, 0);
  Eigen::Vector2d C(0, 0);

  parametricbem2d::ParametrizedLine l1(C, R); // right
  parametricbem2d::ParametrizedLine l2(B, C); // top
  parametricbem2d::ParametrizedCircularArc curve(Eigen::Vector2d(0,0),1,0,3*M_PI/2);

  unsigned order = 16;
  std::cout << "#quadrature order: " << order << std::endl;
  std::cout << std::setw(10) << "#numpanels" << std::setw(25) << "c*(gradu.n)^2"
            << std::setw(25) << "BEM" << std::setw(25) << "0.5*(gradu)^2 ex."
            << std::setw(25) << "Boundary Formula 1" << std::setw(25)
            << "Boundary Formula 2" << std::endl;

  out << "#quadrature order: " << order << std::endl;
  out << std::setw(10) << "#numpanels" << std::setw(25) << "c*(gradu.n)^2"
            << std::setw(25) << "BEM" << std::setw(25) << "0.5*(gradu)^2 ex."
            << std::setw(25) << "Boundary Formula 1" << std::setw(25)
            << "Boundary Formula 2" << std::endl;

  for (unsigned numpanels = 2; numpanels < 3; numpanels += 1) {
    //auto start = std::chrono::system_clock::now();
    unsigned temp = numpanels;
    parametricbem2d::PanelVector panels_l1(l1.split(temp));
    parametricbem2d::PanelVector panels_curve(curve.split(temp));
    parametricbem2d::PanelVector panels_l2(l2.split(temp));

    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(), panels_l1.begin(), panels_l1.end());
    panels.insert(panels.end(), panels_curve.begin(), panels_curve.end());
    panels.insert(panels.end(), panels_l2.begin(), panels_l2.end());
    parametricbem2d::ParametrizedMesh mesh(panels);

    double force = CalculateForce(mesh, g, nu, order, out);

    //auto end = std::chrono::system_clock::now();
    //auto elapsed =
    //std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //std::cout << "numpanels: " << mesh.getNumPanels() << "  "<< elapsed.count() << std::endl;
  }

  return 0;
}
