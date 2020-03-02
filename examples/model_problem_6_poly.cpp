#include "BoundaryMesh.hpp"
#include "force_calculation.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

// format of the gradient:
// dv1/dx1  dv2/dx1
// dv1/dx2  dv2/dx2

// format of dgrad:
// d2 v/dx1 dx1              d2 v/ dx1 dx2
// d2 v/ dx1 dx2             d2 v/ dx2 dx2

double ri = 1.5; // Radius of inner cirlce
double ro = 3;   // Radius of outer circle

/*
 * Boolean function which indicates whether a point (x,y) is inside the domain
 */
bool inner(double x, double y) { return (x * x + y * y < 4); }

/*
 * Class representing the fourier boundary conditions
 */
class G_FOURIER {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double phi = atan(y / x);
    double r = sqrt(x * x + y * y);
    if (inner(x, y))
      // return 0.5*(3+cos(2*phi)+2*sin(phi)); // WRONG APPROACH!!!!
      return 2 - y * y / (x * x + y * y) + y / r;
    else
      return 0;
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double phi = atan(y / x);
    double r = sqrt(x * x + y * y);
    double r3 = r * r * r;
    double r4 = r * r3;
    double r5 = r * r4;
    if (inner(x, y))
      return Eigen::Vector2d(x * y * (2 * y / r4 - 1 / r3),
                             (x * x * x * x + x * x * y * (y - 2 * r)) / r5);
    else
      return Eigen::Vector2d(0, 0);
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

/*
 * Class representing the true soultion, computed using Mathematica
 */
class U {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return 2.3774437510817346 +
           (y * (2. / (x * x + y * y) - 0.2222222222222222)) +
           (-1 + (2 * x * x) / (x * x + y * y)) *
               (1.2 / (x * x + y * y) -
                0.014814814814814815 * (x * x + y * y)) -
           2.164042561333445 * log(sqrt(x * x + y * y));
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d(
        (-0.02962962962962963 * pow(x, 11) +
         pow(x, 9) * (-2.164042561333445 - 0.14814814814814814 * pow(y, 2)) +
         pow(x, 5) * pow(y, 2) *
             (2.3999999999999995 + y * (-12. - 12.984255368000667 * y -
                                        0.2962962962962963 * pow(y, 3))) +
         pow(x, 7) * (-2.4 + y * (-4. - 8.65617024533378 * y -
                                  0.2962962962962963 * pow(y, 3))) +
         pow(x, 3) * pow(y, 4) *
             (11.999999999999998 + y * (-12. - 8.65617024533378 * y -
                                        0.14814814814814814 * pow(y, 3))) +
         x * pow(y, 6) *
             (7.199999999999999 + y * (-4. - 2.164042561333445 * y -
                                       0.02962962962962963 * pow(y, 3)))) /
            pow(pow(x, 2) + pow(y, 2), 5),
        (0. + pow(x, 10) * (-0.2222222222222222 + 0.02962962962962963 * y) +
         2.4 * pow(y, 7) - 2. * pow(y, 8) - 2.164042561333445 * pow(y, 9) -
         0.2222222222222222 * pow(y, 10) + 0.02962962962962963 * pow(y, 11) +
         pow(x, 8) *
             (2. + y * (-2.164042561333445 +
                        (-1.111111111111111 + 0.14814814814814814 * y) * y)) +
         pow(x, 4) * pow(y, 3) *
             (-11.999999999999998 +
              pow(y, 2) * (-12.984255368000667 +
                           (-2.222222222222222 + 0.2962962962962963 * y) * y)) +
         pow(x, 2) * pow(y, 5) *
             (-2.3999999999999995 +
              y * (-4. +
                   y * (-8.65617024533378 +
                        (-1.111111111111111 + 0.14814814814814814 * y) * y))) +
         pow(x, 6) * y *
             (-7.199999999999999 +
              y * (4. +
                   y * (-8.65617024533378 +
                        (-2.222222222222222 + 0.2962962962962963 * y) * y)))) /
            pow(pow(x, 2) + pow(y, 2), 5));
  }
};

int main() {
  G_FOURIER g; // dirichlet boundary conditions
  U u;         // True solution

  // Initializing the variables m,n (used in the velocity fields) using
  // environment variables MM, NN
  unsigned m = MM;
  unsigned n = NN;

// Velocity field is in the x direction
#if VEL == 1
  std::string filename("mp6polyxymn1_");
  filename += to_string(m) + "_" + to_string(n);

  std::ofstream out(filename);
  std::cout << "annular circle fourier, nu_xymn_1" << std::endl;
  out << "#annular circle fourier, nu_xymn_1" << std::endl;
  NU_XYMN_1<MM, NN> nu;
#endif

// Velocity field y direction
#if VEL == 2
  std::string filename("mp6polyxymn2_");
  filename += to_string(m) + "_" + to_string(n);

  std::ofstream out(filename);
  std::cout << "annular circle fourier, nu_xymn_2" << std::endl;
  out << "#annular circle fourier, nu_xymn_2" << std::endl;
  NU_XYMN_2<MM, NN> nu;
#endif

  std::cout << "#MM NN: " << MM << " " << NN << std::endl;

  // Definition of the domain
  std::cout << "#concentric circles simon" << std::endl;
  parametricbem2d::ParametrizedCircularArc outer(Eigen::Vector2d(0, 0), ro, 0,
                                                 2 * M_PI);
  parametricbem2d::ParametrizedCircularArc inner(Eigen::Vector2d(0, 0), ri,
                                                 2 * M_PI, 0);
  // quadrature order
  unsigned order = 32;
  std::cout << "#quadrature order: " << order << std::endl;
  std::cout << std::setw(10) << "#numpanels" << std::setw(25) << "c*(gradu.n)^2"
            << std::setw(25) << "BEM" << std::setw(25) << "0.5*(gradu)^2 ex."
            << std::setw(25) << "Boundary Formula 1" << std::setw(25)
            << "Boundary Formula 2" << std::endl;
  for (unsigned numpanels = 4; numpanels < 1000; numpanels += 3) {
    unsigned numpanels_i = numpanels; // # panels on the inner boundary
    unsigned numpanels_o = numpanels; // # panels on the outer boundary
    // Getting the panels
    parametricbem2d::PanelVector panels_i = inner.split(numpanels_i);
    parametricbem2d::PanelVector panels_o = outer.split(numpanels_o);
    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(), panels_i.begin(),
                  panels_i.end()); // inner panels added first
    panels.insert(panels.end(), panels_o.begin(), panels_o.end());
    // Initializing the ParametricMesh object
    parametricbem2d::ParametrizedMesh mesh(panels);
    // Evaluating the shape gradient formuulas
    double force = CalculateForce(mesh, g, nu, order, out, u);
  }

  return 0;
}
