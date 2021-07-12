//#include "BoundaryMesh.hpp"
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
 * Class to specify the dirichlet data
 */
class G_CONST {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
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

Eigen::Vector2d exkite(double t) {
  return Eigen::Vector2d(0.3+.35 * std::cos(t) + .1625 * std::cos(2 * t),
                         0.5+.35 * std::sin(t));
}

Eigen::Vector2d exdkite(double t) {
  return Eigen::Vector2d(-.35 * std::sin(t) - 2 * .1625 * std::sin(2 * t),
                         .35 * std::cos(t));
}

Eigen::VectorXd get_kite_params(unsigned N) {
  // Calculating the length of the kite
  unsigned N_length = 500; // No. of points used in the calculation
  Eigen::VectorXd pts_length = Eigen::VectorXd::LinSpaced(N_length,0,2*M_PI);
  double L = 0;
  for (unsigned i = 0 ; i < N_length-1 ; ++i)
    L += (exkite(pts_length(i)) - exkite(pts_length(i+1))).norm();

  std::cout << "found length of the kite: " << L << std::endl;
  // Solving the equation for Phi using explicit timestepping
  unsigned k = 20; // multiplicity factor?
  double h = L/N/k; // step size
  Eigen::VectorXd phi_full = Eigen::VectorXd::Constant(N*k,0);
  Eigen::VectorXd phi = Eigen::VectorXd::Constant(N,0);
  for (unsigned i = 1 ; i < N*k ; ++i)
    phi_full(i) = phi_full(i-1) + h /( exdkite(phi_full(i-1)) ).norm();

  for (unsigned i = 0 ; i < N ; ++i)
    phi(i) = phi_full(i*k);

  return phi;
}


/*
 * Class for nu(x,y,m,n) = [sin(mx) cos(ny), 0]
 */
template <int m, int n> class NU_XYMN_1 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::Vector2d(0,0);
    return Eigen::Vector2d(sin(m * x) * sin(n * y), 0);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::MatrixXd::Constant(2,2,0);
    Eigen::MatrixXd M(2, 2);
    M << m * cos(m * x) * sin(n * y), 0, n * cos(n * y) * sin(m * x), 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return 0;
    return m * cos(m * x) * sin(n * y);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::MatrixXd::Constant(2,2,0);
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
 * Class for nu(x,y,m,n) = [0, sin(mx) cos(ny)]
 */
template <int m, int n> class NU_XYMN_2 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::Vector2d(0,0);
    Eigen::Vector2d out;
    return Eigen::Vector2d(0, sin(m * x) * sin(n * y));
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::MatrixXd::Constant(2,2,0);
    Eigen::MatrixXd M(2, 2);
    M << 0, m * cos(m * x) * sin(n * y), 0, n * cos(n * y) * sin(m * x);
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return 0;
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
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::MatrixXd::Constant(2,2,0);
    Eigen::MatrixXd M(2, 2);
    M << -m * m * sin(m * x) * sin(n * y), m * n * cos(m * x) * cos(n * y),
        m * n * cos(m * x) * cos(n * y), -n * n * sin(m * x) * sin(n * y);
    return M;
  }
};

int main() {
  G_CONST g;
  // Initializing m,n (in velocity fields) from the environment
  // variables MM, NN
  unsigned m = MM;
  unsigned n = NN;

// Velocity field in x direction
#if VEL == 1
  std::string filename("sqkt_sin_1_");
  filename += to_string(m) + "_" + to_string(n);

  std::ofstream out(filename);
  std::cout << "square and square, nu_xymn_1" << std::endl;
  out << "#square and square, nu_xymn_1" << std::endl;
  NU_XYMN_1<MM, NN> nu;
#endif

// Velocity field in y direction
#if VEL == 2
  std::string filename("sqkt_sin_2_");
  filename += to_string(m) + "_" + to_string(n);

  std::ofstream out(filename);
  std::cout << "square and square, nu_xymn_2" << std::endl;
  out << "#square and square, nu_xymn_2" << std::endl;
  NU_XYMN_2<MM, NN> nu;
#endif

  // Defining the kite domain
  Eigen::MatrixXd cos_list_o(2, 2);
  cos_list_o << .35, .1625, 0, 0;
  Eigen::MatrixXd sin_list_o(2, 2);
  sin_list_o << 0, 0, .35, 0;
  parametricbem2d::ParametrizedFourierSum kite(
      Eigen::Vector2d(0.3, 0.5), cos_list_o, sin_list_o, 2 * M_PI, 0);

  unsigned order = 16;
  // Calculating the center of mass for the inner body:
  auto cg = [&](parametricbem2d::PanelVector& panels) {
    unsigned N = panels.size();
    double integralx(0), integraly(0), integrald(0);
    for (unsigned i = 0 ; i < N ; ++i) {
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      // X component
      auto integrandx = [&](double t) {
        double x = pi(t)(0);
        Eigen::Vector2d tangent = pi.Derivative(t);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return Eigen::Vector2d(x*x/2.,0).dot(normal) * pi.Derivative(t).norm();
      };
      // Y component
      auto integrandy = [&](double t) {
        double y = pi(t)(1);
        Eigen::Vector2d tangent = pi.Derivative(t);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return Eigen::Vector2d(0,y*y/2.).dot(normal) * pi.Derivative(t).norm();
      };
      // Denominator
      auto integrandd = [&](double t) {
        double x = pi(t)(0);
        Eigen::Vector2d tangent = pi.Derivative(t);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return Eigen::Vector2d(x,0).dot(normal) * pi.Derivative(t).norm();
      };
      integralx += parametricbem2d::ComputeIntegral(integrandx,-1,1,order);
      integraly += parametricbem2d::ComputeIntegral(integrandy,-1,1,order);
      integrald += parametricbem2d::ComputeIntegral(integrandd,-1,1,order);
    }
    return Eigen::Vector2d(integralx/integrald,integraly/integrald);
  };

  // Velocity field rotational
  #if VEL == 3
    std::string filename("sqkg3_");
    filename += to_string(m) + "_" + to_string(n);

    std::ofstream out(filename);
    std::cout << "square and square, nu_xymn_2" << std::endl;
    out << "#square and square, nu_xymn_2" << std::endl;
    parametricbem2d::PanelVector cgpanels(kite.split(1));
    Eigen::Vector2d CG = cg(cgpanels);
    NU_ROT nu(CG);
    std::cout << "Found CG: " << CG << std::endl;
    std::cout << "nu cg check : " << nu(CG).transpose() << std::endl;
  #endif

  std::cout << "#MM NN: " << MM << " " << NN << std::endl;
  out << "#MM NN: " << MM << " " << NN << std::endl;

  // Defining the outer big square with g = 0
  // Outer square vertices
  Eigen::Vector2d NEo(2, 2);
  Eigen::Vector2d NWo(-2, 2);
  Eigen::Vector2d SEo(2, -2);
  Eigen::Vector2d SWo(-2, -2);
  // Outer square edges
  parametricbem2d::ParametrizedLine Or(SEo, NEo); // right
  parametricbem2d::ParametrizedLine ot(NEo, NWo); // top
  parametricbem2d::ParametrizedLine ol(NWo, SWo); // left
  parametricbem2d::ParametrizedLine ob(SWo, SEo); // bottom
  // quadrature order
  //unsigned order = 16;
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
    unsigned temp = 2*numpanels;
    // Creating the ParametricMesh object
    //parametricbem2d::PanelVector panels_kite(kite.split(numpanels));
    parametricbem2d::PanelVector panels_kite;
    double lkite= 2.46756;
    double lsquare = 16;
    Eigen::VectorXd meshpts = get_kite_params(numpanels);
      unsigned N = meshpts.size();
      Eigen::VectorXd tempp(N+1);
      tempp << meshpts, 2*M_PI;
      tempp = -tempp + Eigen::VectorXd::Constant(N+1,2*M_PI);
      std::cout << "temp: " << tempp.transpose() << std::endl;
      for (unsigned i = 0 ; i < N ; ++i) {
        // Defining the kite domain
        Eigen::MatrixXd cos_list_o(2, 2);
        cos_list_o << .35, .1625, 0, 0;
        Eigen::MatrixXd sin_list_o(2, 2);
        sin_list_o << 0, 0, .35, 0;
        panels_kite.push_back(std::make_shared<parametricbem2d::ParametrizedFourierSum>(
            Eigen::Vector2d(0.3, 0.5), cos_list_o, sin_list_o, tempp(i), tempp(i+1)));
      }

    // Meshing the sqkite equivalently in the parameter mesh
    // Creating the ParametricMesh object
    parametricbem2d::PanelVector panels;
    // Panels for the edges of outer square
    parametricbem2d::PanelVector panels_or(Or.split(temp));
    parametricbem2d::PanelVector panels_ot(ot.split(temp));
    parametricbem2d::PanelVector panels_ol(ol.split(temp));
    parametricbem2d::PanelVector panels_ob(ob.split(temp));

    panels.insert(panels.end(), panels_kite.begin(), panels_kite.end());
    panels.insert(panels.end(), panels_or.begin(), panels_or.end());
    panels.insert(panels.end(), panels_ot.begin(), panels_ot.end());
    panels.insert(panels.end(), panels_ol.begin(), panels_ol.end());
    panels.insert(panels.end(), panels_ob.begin(), panels_ob.end());

    parametricbem2d::ParametrizedMesh mesh(panels);
    // Evaluating the shape gradients; exact solution u not available
    double force = CalculateForce(mesh, g, nu, order, out);
  }

  return 0;
}
