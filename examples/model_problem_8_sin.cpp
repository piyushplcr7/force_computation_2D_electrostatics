#include "BoundaryMesh.hpp"
#include "force_calculation.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>

// format of the gradient:
// dv1/dx1  dv2/dx1
// dv1/dx2  dv2/dx2

// format of dgrad:
// d2 v/dx1 dx1              d2 v/ dx1 dx2
// d2 v/ dx1 dx2             d2 v/ dx2 dx2

double R = 1.5; // 2;

/*
 * Class representing the velocity field nu(x,y,m,n) = [sin(mx) sin(ny), 0]
 */
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

/*
 * Class representing the velocity field nu(x,y,m,n) = [0, sin(mx) sin(ny)]
 */
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

/*
 * Class representing the logarithmic boundary conditions and also the true
 * solution for the potential u
 */
class G {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return log(sqrt(x * x + y * y));
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d(x / (x * x + y * y), y / (x * x + y * y));
  }
};

/*
 * Class representing the boundary conditions imposed by the known logarithmic
 * solution. The boundary conditions are specified in a manner that only changes
 * the gradients at the boundary.
 */
class GC {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    if (r < 1.1 * R)
      return log(R);
    else
      return log(r) + std::pow(x - 2 * R, 2) + std::pow(y, 2) - 16 * R * R;
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    if (r < 1.1 * R)
      return Eigen::Vector2d(0, 0);
    else
      return Eigen::Vector2d(x / (x * x + y * y) + 2 * (x - 2 * R),
                             y / (x * x + y * y) + 2 * y);
  }
};

int main() {
  // Boundary condition
  G g;
  // GC g;
  // True solution
  G u;

  // Initializing the variables m,n (used in the velocity fields) using
  // environment variables MM, NN
  unsigned m = MM;
  unsigned n = NN;

// Velocity field in x direction
#if VEL == 1
  std::string filename("mp8sinxymn1_");
  filename += to_string(m) + "_" + to_string(n);

  std::ofstream out(filename);
  std::cout << "CONC, nu_xymn_1" << std::endl;
  out << "#CONC, nu_xymn_1" << std::endl;
  NU_XYMN_1<MM, NN> nu;
#endif

// Velocity field in y direction
#if VEL == 2
  std::string filename("mp8sinxymn2_");
  filename += to_string(m) + "_" + to_string(n);

  std::ofstream out(filename);
  std::cout << "CONC, nu_xymn_2" << std::endl;
  out << "#CONC, nu_xymn_2" << std::endl;
  NU_XYMN_2<MM, NN> nu;
#endif

  std::cout << "#MM NN: " << MM << " " << NN << std::endl;
  out << "#MM NN: " << MM << " " << NN << std::endl;

  std::cout << "#g logr" << std::endl;
  out << "#g logr" << std::endl;

  // Definition of the domain
  std::cout << "#Shifted circle" << std::endl;
  parametricbem2d::ParametrizedCircularArc outer(Eigen::Vector2d(2 * R, 0),
                                                 4 * R, 0, 2 * M_PI);
  parametricbem2d::ParametrizedCircularArc inner(Eigen::Vector2d(0, 0), R,
                                                 2 * M_PI, 0);

  // Quadrature order
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

  for (unsigned numpanels = 4; numpanels < 1000; numpanels += 3) {
    unsigned numpanels_i = numpanels; // # panels for inner boundary
    unsigned numpanels_o = numpanels; // # panels for outer boundary
    // Getting the panels for creating the ParametricMesh object
    parametricbem2d::PanelVector panels_i = inner.split(numpanels_i);
    parametricbem2d::PanelVector panels_o = outer.split(numpanels_o);
    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(), panels_i.begin(),
                  panels_i.end()); // inner panels added first
    panels.insert(panels.end(), panels_o.begin(), panels_o.end());
    parametricbem2d::ParametrizedMesh mesh(panels);
    // Getting the shape gradients
    double force = CalculateForce(mesh, g, nu, order, out, u);
  }

  return 0;
}
