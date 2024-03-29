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

class G_CONST {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (inner(x, y))
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

class NU_1 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (inner(x,y))
      return Eigen::Vector2d(1, 0);
    else
      return Eigen::Vector2d(0,0);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
      return 0;
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_2 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (inner(x,y))
      return Eigen::Vector2d(0, 1);
    else
      return Eigen::Vector2d(0,0);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
      return 0;
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
};

int main() {
  G_CONST g;

  #if VEL == 1
  std::string filename("mp5fx");

  std::ofstream out(filename);
  std::cout << "square and square, nu_1" << std::endl;
  out << "#square and square, nu_1" << std::endl;
  NU_1 nu;
  #endif

  #if VEL == 2
  std::string filename("mp5fy");

  std::ofstream out(filename);
  std::cout << "square and square, nu_2" << std::endl;
  out << "#square and square, nu_2" << std::endl;
  NU_2 nu;
  #endif

  std::cout << "#Linear Mesh!" << std::endl;
  std::cout << "#g const" << std::endl;
  out << "#Linear Mesh!" << std::endl;
  out << "#g const" << std::endl;

  // Square and Square
  // Inner vertices
  Eigen::Vector2d NE(1, 1);
  Eigen::Vector2d NW(0, 1);
  Eigen::Vector2d SE(1, 0);
  Eigen::Vector2d SW(0, 0);
  // Outer vertices
  Eigen::Vector2d NEo(3, 3);
  Eigen::Vector2d NWo(-3, 3);
  Eigen::Vector2d SEo(3, -3);
  Eigen::Vector2d SWo(-3, -3);
  // Inner square
  parametricbem2d::ParametrizedLine ir(NE, SE); // right
  parametricbem2d::ParametrizedLine it(NW, NE); // top
  parametricbem2d::ParametrizedLine il(SW, NW); // left
  parametricbem2d::ParametrizedLine ib(SE, SW); // bottom
  // Outer Square
  parametricbem2d::ParametrizedLine Or(SEo, NEo); // right
  parametricbem2d::ParametrizedLine ot(NEo, NWo); // top
  parametricbem2d::ParametrizedLine ol(NWo, SWo); // left
  parametricbem2d::ParametrizedLine ob(SWo, SEo); // bottom

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
    //auto start = std::chrono::system_clock::now();
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
    panels.insert(panels.end(), panels_ir.begin(), panels_ir.end());
    panels.insert(panels.end(), panels_ib.begin(), panels_ib.end());
    panels.insert(panels.end(), panels_il.begin(), panels_il.end());
    panels.insert(panels.end(), panels_it.begin(), panels_it.end());

    panels.insert(panels.end(), panels_or.begin(), panels_or.end());
    panels.insert(panels.end(), panels_ot.begin(), panels_ot.end());
    panels.insert(panels.end(), panels_ol.begin(), panels_ol.end());
    panels.insert(panels.end(), panels_ob.begin(), panels_ob.end());
    parametricbem2d::ParametrizedMesh mesh(panels);

    double force = CalculateForce(mesh, g, nu, order, out);
  }

  return 0;
}
