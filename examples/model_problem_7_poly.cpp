#include "BoundaryMesh.hpp"
#include "force_calculation.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>

// format of the gradient:
// dv1/dx1  dv2/dx1
// dv1/dx2  dv2/dx2

// format of dgrad:
// d2 v/dx1 dx1              d2 v/ dx1 dx2
// d2 v/ dx1 dx2             d2 v/ dx2 dx2

/*
 * Class representing the Dirichlet boundary conditions given by the corner
 * singular function. This class contains the singularity.
 */
class G {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    double phi = xy_to_phi(x, y);
    return std::pow(r, 2. / 3.) * sin(2. / 3. * phi);
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    double phi = xy_to_phi(x, y);
    return 2. / 3. * std::pow(r, -1. / 3.) *
           Eigen::Vector2d(-sin(phi / 3.), cos(phi / 3.));
  }
};

/*
 * Class representing the Dirichlet boundary conditions given by the corner
 * singular function. This class does not contain a singularity.
 */
class GS {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    double phi = xy_to_phi(x, y);
    return r * sin(2. / 3. * phi);
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    double phi = xy_to_phi(x, y);
    return Eigen::Vector2d(sin(2. * phi / 3.) * cos(phi) -
                               2. / 3. * cos(2. / 3. * phi) * sin(phi),
                           sin(2. * phi / 3.) * sin(phi) +
                               2. / 3. * cos(2. / 3. * phi) * cos(phi));
  }
};

/*
 * Class representing the Dirichlet boundary conditions given by the corner
 * singular function. This class contains a different singularity than the
 * corner singular function.
 */
class GNS {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    double phi = xy_to_phi(x, y);
    return std::pow(r, 3. / 4.) * sin(2. / 3. * phi);
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    double phi = xy_to_phi(x, y);
    return std::pow(r, -1. / 4.) *
           Eigen::Vector2d(3. / 4. * sin(2. * phi / 3.) * cos(phi) -
                               2. / 3. * cos(2. / 3. * phi) * sin(phi),
                           3. / 4. * sin(2. * phi / 3.) * sin(phi) +
                               2. / 3. * cos(2. / 3. * phi) * cos(phi));
  }
};

/*
 * Class representing the velocity field nu(x,y,m,n) = [x^m * y^n, 0]
 */
template <int m, int n> class NU_XYMN_1 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    return Eigen::Vector2d(std::pow(x, m) * std::pow(y, n), 0);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    double powxm1, powym1;
    if (m == 0)
      powxm1 = 0;
    else
      powxm1 = m * std::pow(x, m - 1);

    if (n == 0)
      powym1 = 0;
    else
      powym1 = n * std::pow(y, n - 1);
    M << powxm1 * std::pow(y, n), 0, std::pow(x, m) * powym1, 0;

    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (m == 0)
      return 0;
    else
      return m * std::pow(x, m - 1) * std::pow(y, n);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    double M11, M12, M22;

    if (m == 0 or m == 1)
      M11 = 0;
    else
      M11 = m * (m - 1) * std::pow(x, m - 2) * std::pow(y, n);

    if (m == 0 or n == 0)
      M12 = 0;
    else
      M12 = m * n * std::pow(x, m - 1) * std::pow(y, n - 1);

    if (n == 0 or n == 1)
      M22 = 0;
    else
      M22 = n * (n - 1) * std::pow(x, m) * std::pow(y, n - 2);

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

/*
 * Class representing the velocity field nu(x,y,m,n) = [0, x^m * y^n]
 */
template <int m, int n> class NU_XYMN_2 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    return Eigen::Vector2d(0, std::pow(x, m) * std::pow(y, n));
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    double powxm1, powym1;
    if (m == 0)
      powxm1 = 0;
    else
      powxm1 = m * std::pow(x, m - 1);

    if (n == 0)
      powym1 = 0;
    else
      powym1 = n * std::pow(y, n - 1);
    M << 0, powxm1 * std::pow(y, n), 0, std::pow(x, m) * powym1;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (n == 0)
      return 0;
    else
      return n * std::pow(x, m) * std::pow(y, n - 1);
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
    double M11, M12, M22;

    if (m == 0 or m == 1)
      M11 = 0;
    else
      M11 = m * (m - 1) * std::pow(x, m - 2) * std::pow(y, n);

    if (m == 0 or n == 0)
      M12 = 0;
    else
      M12 = m * n * std::pow(x, m - 1) * std::pow(y, n - 1);

    if (n == 0 or n == 1)
      M22 = 0;
    else
      M22 = n * (n - 1) * std::pow(x, m) * std::pow(y, n - 2);

    M << M11, M12, M12, M22;
    return M;
  }
};

int main() {
  // Boundary condition
  // G g;
  // GS g;
  GNS g;

  // Initializing the variables m,n (used in the velocity fields) using
  // environment variables MM, NN
  unsigned m = MM;
  unsigned n = NN;

// Velocity field in x direction
#if VEL == 1
  std::string filename("mp7polyxymn1_");
  filename += to_string(m) + "_" + to_string(n);

  std::ofstream out(filename);
  std::cout << "pacman, nu_xymn_1" << std::endl;
  out << "#pacman, nu_xymn_1" << std::endl;
  NU_XYMN_1<MM, NN> nu;
#endif

// Velocity field in y direction
#if VEL == 2
  std::string filename("mp7polyxymn2_");
  filename += to_string(m) + "_" + to_string(n);

  std::ofstream out(filename);
  std::cout << "pacman, nu_xymn_2" << std::endl;
  out << "#pacman, nu_xymn_2" << std::endl;
  NU_XYMN_2<MM, NN> nu;
#endif

  std::cout << "#MM NN: " << MM << " " << NN << std::endl;
  out << "#MM NN: " << MM << " " << NN << std::endl;

  // Definition of the pacman domain
  Eigen::Vector2d B(0, -1); // Bottom most point
  Eigen::Vector2d R(1, 0);  // Right most point
  Eigen::Vector2d C(0, 0);  // Center

  parametricbem2d::ParametrizedLine l1(C, R); // Edge parallel to x axis
  parametricbem2d::ParametrizedLine l2(B, C); // Edge parallel to y axis
  // Curved part of the boundary
  parametricbem2d::ParametrizedCircularArc curve(Eigen::Vector2d(0, 0), 1, 0,
                                                 3 * M_PI / 2);
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

  for (unsigned numpanels = 2; numpanels < 1001; numpanels += 1) {
    unsigned temp = numpanels;
    // Getting the panels for different parts of the boundary
    parametricbem2d::PanelVector panels_l1(l1.split(temp));
    parametricbem2d::PanelVector panels_curve(curve.split(temp));
    parametricbem2d::PanelVector panels_l2(l2.split(temp));
    // Creating the ParametricMesh object
    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(), panels_l1.begin(), panels_l1.end());
    panels.insert(panels.end(), panels_curve.begin(), panels_curve.end());
    panels.insert(panels.end(), panels_l2.begin(), panels_l2.end());
    parametricbem2d::ParametrizedMesh mesh(panels);
    // Evaluating the shape gradients
    double force = CalculateForce(mesh, g, nu, order, out, u);
  }

  return 0;
}
