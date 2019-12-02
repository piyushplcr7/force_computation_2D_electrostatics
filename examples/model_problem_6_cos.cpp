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

bool inner(double x, double y) {
  return (x*x+y*y<4);
}

template <int m, int n> class NU_XYMN_1 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    return Eigen::Vector2d(cos(m * x) * cos(n * y), 0);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << -m * sin(m * x) * cos(n * y), 0, -n * sin(n * y) * cos(m * x), 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return -m * sin(m * x) * cos(n * y);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << -m * m * cos(m * x) * cos(n * y), m * n * sin(m * x) * sin(n * y),
        m * n * sin(m * x) * sin(n * y), -n * n * cos(m * x) * cos(n * y);
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
    return Eigen::Vector2d(0, cos(m * x) * cos(n * y));
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 0, -m * sin(m * x) * cos(n * y), 0, -n * sin(n * y) * cos(m * x);
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return -n * sin(n * y) * cos(m * x);
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
    M << -m * m * cos(m * x) * cos(n * y), m * n * sin(m * x) * sin(n * y),
        m * n * sin(m * x) * sin(n * y), -n * n * cos(m * x) * cos(n * y);
    return M;
  }
};

class G_FOURIER {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double phi = atan(y/x);
    double r = sqrt(x*x+y*y);
    if (inner(x,y))
      //return 0.5*(3+cos(2*phi)+2*sin(phi)); // WRONG APPROACH!!!!
      return 2-y*y/(x*x+y*y)+y/r;
    else
      return 0;
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double phi = atan(y/x);
    double r = sqrt(x*x+y*y);
    double r3 = r*r*r;
    double r4 = r*r3;
    double r5 = r*r4;
    if (inner(x,y))
      return Eigen::Vector2d(x*y*(2*y/r4-1/r3),(x*x*x*x+x*x*y*(y-2*r))/r5);
    else
      return Eigen::Vector2d(0,0);
  }
};

int main() {
  G_FOURIER g;

  unsigned m = MM;
  unsigned n = NN;

  #if VEL == 1
  std::string filename("mp6cosxymn1_");
  filename += to_string(m)+"_"+to_string(n);

  std::ofstream out(filename);
  std::cout << "annular circle fourier, nu_xymn_1" << std::endl;
  out << "#annular circle fourier, nu_xymn_1" << std::endl;
  NU_XYMN_1<MM,NN> nu;
  #endif

  #if VEL == 2
  std::string filename("mp6cosxymn2_");
  filename += to_string(m)+"_"+to_string(n);

  std::ofstream out(filename);
  std::cout << "annular circle fourier, nu_xymn_2" << std::endl;
  out << "#annular circle fourier, nu_xymn_2" << std::endl;
  NU_XYMN_2<MM,NN> nu;
  #endif

  std::cout << "#MM NN: " << MM << " " << NN << std::endl;

  // Inner circle Outer kite
  std::cout << "#concentric circles simon" << std::endl;
  parametricbem2d::ParametrizedCircularArc outer(Eigen::Vector2d(0,0),ro,0,2*M_PI);
  parametricbem2d::ParametrizedCircularArc inner(Eigen::Vector2d(0,0),ri,2*M_PI,0);
  //parametricbem2d::ParametrizedFourierSum inner(
  //    Eigen::Vector2d(0., 0.), cos_list_i, sin_list_i, 2 * M_PI, 0);

  unsigned order = 32;
  std::cout << "#quadrature order: " << order << std::endl;
  std::cout << std::setw(10) << "#numpanels" << std::setw(25) << "c*(gradu.n)^2"
            << std::setw(25) << "BEM" << std::setw(25) << "0.5*(gradu)^2 ex."
            << std::setw(25) << "Boundary Formula 1" << std::setw(25)
            << "Boundary Formula 2" << std::endl;
  for (unsigned numpanels = 4; numpanels < 1000; numpanels += 3) {
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
    //parametricbem2d::ParametrizedMesh lmesh = convert_to_linear(mesh);
    double force = CalculateForce(mesh, g, nu, order,out);
    // Force using linear mesh!
    //double force = CalculateForce(lmesh, g, nu, order);
  }

  return 0;
}
