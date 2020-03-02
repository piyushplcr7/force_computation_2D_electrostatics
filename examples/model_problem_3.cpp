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

// Global variable for the radius
double R = 2;
// x coordinate of the center for the outer circle
double K = 0;

/*
 * Class representing the modified radial velocity field
 */
class NU_RADIAL_APP {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    if (x * x + y * y < 1.1 * 1.1 * R * R) // Inner circle
      out << x + x * x + y * y - R * R, y + x * x + y * y - R * R;
    else // Outer circle
      out << x + x * x + y * y - 4 * R * 4 * R,
          y + x * x + y * y - 4 * R * 4 * R;
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 1 + 2 * x, 2 * x, 2 * y, 1 + 2 * y;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return 2 * (1 + x + y);
  }
  // Second order derivative of the velocity field's first component
  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 2, 0, 0, 2;
    return M;
  }
  // Second order derivative of the velocity field's second component
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 2, 0, 0, 2;
    return M;
  }
};

/*
 * Class specifying the dirichlet data and the true potential u
 */
class G {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return x + y;
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d(1, 1);
  }
};

/*
 * Class representing the normalized radial velocity field
 */
class NU_RADIAL_NORM {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    Eigen::Vector2d out;
    out << x / r, y / r;
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r3 = r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << y * y / r3, -x * y / r3, -x * y / r3, x * x / r3;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    return 1 / r;
  }
  // Second order derivative of the velocity field's first component
  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r5 = r * r * r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << -3 * x * y * y / r5, y * (2 * x * x - y * y) / r5,
        y * (2 * x * x - y * y) / r5, x * (2 * y * y - x * x) / r5;
    return M;
  }
  // Second order derivative of the velocity field's second component
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r5 = r * r * r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << y * (2 * x * x - y * y) / r5, x * (2 * y * y - x * x) / r5,
        x * (2 * y * y - x * x) / r5, -3 * x * x * y / r5;
    return M;
  }
};

/*
 * Class representing the modified normalized radial velocity field
 */
class NU_RADIAL_NORM_APP {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    Eigen::Vector2d out;
    out << x / r + x * x + y * y - R * R, y / r + x * x + y * y - R * R;
    if (x * x + y * y < 1.1 * 1.1 * R * R)
      return out;
    else
      return Eigen::Vector2d(x / r + (x - K) * (x - K) + y * y - 16 * R * R,
                             y / r + (x - K) * (x - K) + y * y - 16 * R * R);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r3 = r * r * r;
    Eigen::MatrixXd M(2, 2);
    if (x * x + y * y < 1.1 * 1.1 * R * R)
      M << y * y / r3 + 2 * x, 2 * x - x * y / r3, 2 * y - x * y / r3,
          2 * y + x * x / r3;
    else
      M << y * y / r3 + 2 * (x - K), 2 * (x - K) - x * y / r3,
          2 * y - x * y / r3, 2 * y + x * x / r3;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    if (x * x + y * y < 1.1 * 1.1 * R * R)
      return 1 / r + 2 * x + 2 * y;
    else
      return 1 / r + 2 * (x - K) + 2 * y;
  }
  // Second order derivative of the velocity field's first component
  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r5 = r * r * r * r * r;
    Eigen::MatrixXd M(2, 2);
    if (x * x + y * y < 1.1 * 1.1 * R * R)
      M << 2 - 3 * x * y * y / r5, y * (2 * x * x - y * y) / r5,
          y * (2 * x * x - y * y) / r5, 2 + x * (2 * y * y - x * x) / r5;
    else
      M << 2 - 3 * x * y * y / r5, y * (2 * x * x - y * y) / r5,
          y * (2 * x * x - y * y) / r5, 2 + x * (2 * y * y - x * x) / r5;
    return M;
  }
  // Second order derivative of the velocity field's second component
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r5 = r * r * r * r * r;
    Eigen::MatrixXd M(2, 2);
    if (x * x + y * y < 1.1 * 1.1 * R * R)
      M << 2 + y * (2 * x * x - y * y) / r5, x * (2 * y * y - x * x) / r5,
          x * (2 * y * y - x * x) / r5, 2 - 3 * x * x * y / r5;
    else
      M << 2 + y * (2 * x * x - y * y) / r5, x * (2 * y * y - x * x) / r5,
          x * (2 * y * y - x * x) / r5, 2 - 3 * x * x * y / r5;
    return M;
  }
};

int main() {
  std::string fname = "mp3.txt";
  std::ofstream out(fname);
  G g;
  G u; // u = g
  NU_RADIAL_APP nu;
  // NU_RADIAL_NORM_APP nu;
  // NU_RADIAL_NORM nu;

  std::cout << "#Model problem 3" << std::endl;
  std::cout << "#concentric circles" << std::endl;
  // outer circle
  parametricbem2d::ParametrizedCircularArc outer(Eigen::Vector2d(0, 0), 4 * R,
                                                 0, 2 * M_PI);
  // Inner circle
  parametricbem2d::ParametrizedCircularArc inner(Eigen::Vector2d(0, 0), R,
                                                 2 * M_PI, 0);
  // quadrature order
  unsigned order = 16;
  std::cout << "#quadrature order: " << order << std::endl;
  std::cout << std::setw(10) << "#numpanels" << std::setw(25) << "c*(gradu.n)^2"
            << std::setw(25) << "BEM" << std::setw(25) << "0.5*(gradu)^2 ex."
            << std::setw(25) << "Boundary Formula 1" << std::setw(25)
            << "Boundary Formula 2" << std::endl;
  for (unsigned numpanels = 4; numpanels < 500; numpanels += 3) {
    unsigned numpanels_i = numpanels; // # panels for inner boundary
    unsigned numpanels_o = numpanels; // # panels for outer boundary
    // Getting the panels
    parametricbem2d::PanelVector panels_i = inner.split(numpanels_i);
    parametricbem2d::PanelVector panels_o = outer.split(numpanels_o);
    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(), panels_i.begin(),
                  panels_i.end()); // inner panels added first
    panels.insert(panels.end(), panels_o.begin(), panels_o.end());
    // Making the mesh object
    parametricbem2d::ParametrizedMesh mesh(panels);
    // Evaluating shape gradients
    double force = CalculateForce(mesh, g, nu, order, u, out);
  }

  return 0;
}
