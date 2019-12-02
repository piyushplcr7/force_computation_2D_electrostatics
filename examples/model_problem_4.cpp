#include "BoundaryMesh.hpp"
#include "force_calculation.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>

parametricbem2d::ParametrizedMesh
convert_to_linear(const parametricbem2d::ParametrizedMesh &pmesh) {
  unsigned N = pmesh.getNumPanels();
  parametricbem2d::PanelVector panels = pmesh.getPanels();
  parametricbem2d::PanelVector lmesh;
  for (unsigned i = 0; i < N; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    // Making linear panels using the end points of the original panel
    parametricbem2d::ParametrizedLine lpanel(pi(-1), pi(1));
    parametricbem2d::PanelVector tmp = lpanel.split(1);
    lmesh.insert(lmesh.end(), tmp.begin(), tmp.end());
  }
  parametricbem2d::ParametrizedMesh plmesh(lmesh);
  return plmesh;
}

// format of the gradient:
// dv1/dx1  dv2/dx1
// dv1/dx2  dv2/dx2

// format of dgrad:
// d2 v/dx1 dx1              d2 v/ dx1 dx2
// d2 v/ dx1 dx2             d2 v/ dx2 dx2

double ri = 1.5;
double ro = 3;

double R = 2;

bool inner(double x, double y) {
  if ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) < 4)
    return true;
  else
    return false;
}

class NU_CONST_X_INNER {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    if (inner(x,y))
      out << 1, 0;
    else
      out << 0, 0;
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const { return 0; }

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
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_CONST_Y_INNER {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    if (inner(x,y))
      out << 1, 0;
    else
      out << 0, 0;
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const { return 0; }

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
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_RADIAL {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    out << x, y;
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 1, 0, 0, 1;
    return M;
  }

  double div(const Eigen::Vector2d &X) const { return 2; }

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
    M << 0, 0, 0, 0;
    return M;
  }
};

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

class NU_X_INNER_SHIFTED {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    if (inner(x,y))
      out << x-0.5, 0;
    else
      out << 0, 0;
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    double x = X(0);
    double y = X(1);
    if (inner(x,y))
      M << 1, 0, 0, 0;
    else
      M << 0, 0, 0, 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (inner(x,y))
      return 1;
    else
      return 0;
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
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_Y_INNER_SHIFTED {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    if (inner(x,y))
      out << 0, y-0.5;
    else
      out << 0, 0;
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    if (inner(x,y))
      M << 0, 0, 0, 1;
    else
      M << 0, 0, 0, 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (inner(x,y))
      return 1;
    else
      return 0;
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
    M << 0, 0, 0, 0;
    return M;
  }
};

class G_LINEAR {
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

class G_CONST {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (inner(x,y))
      return 0;
    else
      return 1;
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d(0, 0);
  }
};

class G_SHIFTED_LOG {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return log(sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)));
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d(
        (x - 0.5) / ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)),
        (y - 0.5) / ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)));
  }
};

class G_LOG {
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

int main() {
  //G_LINEAR g;
  //G_SHIFTED_LOG g;
  G_LOG g;
  //G_CONST g;

  //NU_CONST_X_INNER nu;
  //NU_CONST_Y_INNER nu;
  //NU_X_INNER_SHIFTED nu;
  //NU_Y_INNER_SHIFTED nu;
  //NU_RADIAL nu;
  NU_RADIAL_NORM nu;

  std::cout << "Linear Mesh!" << std::endl;
  std::cout << "g log, nu radialnorm" << std::endl;
  // Inner circle Outer kite
  std::cout << "#kite and kite" << std::endl;
  Eigen::MatrixXd cos_list_o(2, 2);
  cos_list_o << 3.5, 1.625, 0, 0;
  Eigen::MatrixXd sin_list_o(2, 2);
  sin_list_o << 0, 0, 3.5, 0;
  parametricbem2d::ParametrizedFourierSum outer(
      Eigen::Vector2d(0, 0), cos_list_o, sin_list_o, 0, 2 * M_PI);
  Eigen::MatrixXd cos_list_i(2, 2);
  cos_list_i << -0.8, -0.3, 0, 0;
  Eigen::MatrixXd sin_list_i(2, 2);
  sin_list_i << 0, 0, 1, 0;
  parametricbem2d::ParametrizedFourierSum inner(
      Eigen::Vector2d(0.5, 0.5), cos_list_i, sin_list_i, 0, 2 * M_PI);
  //parametricbem2d::ParametrizedFourierSum inner(
  //    Eigen::Vector2d(0., 0.), cos_list_i, sin_list_i, 2 * M_PI, 0);

  unsigned order = 16;
  std::cout << "#quadrature order: " << order << std::endl;
  std::cout << std::setw(10) << "#numpanels" << std::setw(25) << "c*(gradu.n)^2"
            << std::setw(25) << "BEM" << std::setw(25) << "0.5*(gradu)^2 ex."
            << std::setw(25) << "Boundary Formula 1" << std::setw(25)
            << "Boundary Formula 2" << std::endl;
  for (unsigned numpanels = 4; numpanels < 500; numpanels += 3) {
    // parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
    unsigned numpanels_i = numpanels;
    unsigned numpanels_o = numpanels;
    parametricbem2d::PanelVector panels_i = inner.split(numpanels_i);
    parametricbem2d::PanelVector panels_o = outer.split(numpanels_o);
    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(), panels_i.begin(),
                  panels_i.end()); // inner panels added first
    panels.insert(panels.end(), panels_o.begin(), panels_o.end());
    parametricbem2d::ParametrizedMesh mesh(panels);
    parametricbem2d::ParametrizedMesh lmesh = convert_to_linear(mesh);
    //double force = CalculateForce(mesh, g, nu, order);
    // Force using linear mesh!
    double force = CalculateForce(lmesh, g, nu, order);
  }

  return 0;
}
