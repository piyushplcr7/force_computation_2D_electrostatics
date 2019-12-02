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

bool inner(double x, double y) {
  return (fabs(x)<2 && fabs(y)<2);
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
      out << x, 0;
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
      out << 0, y;
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

template <unsigned m, unsigned n>
class NU_XYMN {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    return Eigen::Vector2d(sin(m*x)*sin(n*y),0);
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

int main() {
  //G_LINEAR g;
  G_CONST g;

  //NU_CONST_X_INNER nu;
  //NU_X_INNER_SHIFTED nu;
  //NU_RADIAL nu;
  NU_RADIAL_NORM nu;
  std::cout << "#Linear Mesh!" << std::endl;
  std::cout << "#g const, nu radialnorm" << std::endl;

  // Square and Square
  std::cout << "Square and Square" << std::endl;
  // Inner vertices
  Eigen::Vector2d NE(1,1);
  Eigen::Vector2d NW(0,1);
  Eigen::Vector2d SE(1,0);
  Eigen::Vector2d SW(0,0);
  // Outer vertices
  Eigen::Vector2d NEo(3,3);
  Eigen::Vector2d NWo(-3,3);
  Eigen::Vector2d SEo(3,-3);
  Eigen::Vector2d SWo(-3,-3);
  // Inner square
  parametricbem2d::ParametrizedLine ir(NE,SE); // right
  parametricbem2d::ParametrizedLine it(NW,NE); // top
  parametricbem2d::ParametrizedLine il(SW,NW); // left
  parametricbem2d::ParametrizedLine ib(SE,SW); // bottom
  // Outer Square
  parametricbem2d::ParametrizedLine Or(SEo,NEo); // right
  parametricbem2d::ParametrizedLine ot(NEo,NWo); // top
  parametricbem2d::ParametrizedLine ol(NWo,SWo); // left
  parametricbem2d::ParametrizedLine ob(SWo,SEo); // bottom

  unsigned order = 16;
  std::cout << "#quadrature order: " << order << std::endl;
  std::cout << std::setw(10) << "#numpanels" << std::setw(25) << "c*(gradu.n)^2"
            << std::setw(25) << "BEM" << std::setw(25) << "0.5*(gradu)^2 ex."
            << std::setw(25) << "Boundary Formula 1" << std::setw(25)
            << "Boundary Formula 2" << std::endl;
  std::cout << "TEST1" << std::endl;
  for (unsigned numpanels = 2000; numpanels < 2001; numpanels += 1) {
    unsigned temp = numpanels;
    parametricbem2d::PanelVector panels_ir(ir.split(temp));
    parametricbem2d::PanelVector panels_it(it.split(temp));
    parametricbem2d::PanelVector panels_il(il.split(temp));
    parametricbem2d::PanelVector panels_ib(ib.split(temp));

    parametricbem2d::PanelVector panels_or(Or.split(temp));
    parametricbem2d::PanelVector panels_ot(ot.split(temp));
    parametricbem2d::PanelVector panels_ol(ol.split(temp));
    parametricbem2d::PanelVector panels_ob(ob.split(temp));

    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(),panels_ir.begin(),panels_ir.end());
    panels.insert(panels.end(),panels_ib.begin(),panels_ib.end());
    panels.insert(panels.end(),panels_il.begin(),panels_il.end());
    panels.insert(panels.end(),panels_it.begin(),panels_it.end());

    panels.insert(panels.end(),panels_or.begin(),panels_or.end());
    panels.insert(panels.end(),panels_ot.begin(),panels_ot.end());
    panels.insert(panels.end(),panels_ol.begin(),panels_ol.end());
    panels.insert(panels.end(),panels_ob.begin(),panels_ob.end());
    parametricbem2d::ParametrizedMesh mesh(panels);

    double force = CalculateForce(mesh, g, nu, order);
  }

  return 0;
}
