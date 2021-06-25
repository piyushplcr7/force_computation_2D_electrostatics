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
 * Class representing the velocity field nu
 */
class NU_ROT {
private:
  Eigen::Vector2d center;
public:
  NU_ROT(const Eigen::Vector2d &x0):center(x0){};

  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::Vector2d(0,0);

    double r = (X-center).norm();
    Eigen::Vector2d xnew = X - center;
    x = xnew(0);
    y = xnew(1);
    return Eigen::Vector2d(-y,x);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      M << 0, 0, 0, 0;
    M << 0,1,-1,0;
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

/*
 * Class representing the velocity field nu(x,y,m,n) = [x^m * y^n, 0]
 */
template <int m, int n> class NU_XYMN_1 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::Vector2d(0,0);
    return Eigen::Vector2d(std::pow(x, m) * std::pow(y, n), 0);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::MatrixXd::Constant(2,2,0);
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
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return 0;
    if (m == 0)
      return 0;
    else
      return m * std::pow(x, m - 1) * std::pow(y, n);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::MatrixXd::Constant(2,2,0);
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
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::Vector2d(0,0);
    return Eigen::Vector2d(0, std::pow(x, m) * std::pow(y, n));
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::MatrixXd::Constant(2,2,0);
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
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return 0;
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
    if (abs(x) > 1.99 || abs(y) > 1.99)
      return Eigen::MatrixXd::Constant(2,2,0);
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
  G_CONST g;
  // Initializing m,n (in velocity fields) from the environment
  // variables MM, NN
  unsigned m = MM;
  unsigned n = NN;

// Velocity field in x direction
#if VEL == 1
  std::string filename("sqkt_bdry_1_");
  filename += to_string(m) + "_" + to_string(n);

  std::ofstream out(filename);
  std::cout << "square and square, nu_xymn_1" << std::endl;
  out << "#square and square, nu_xymn_1" << std::endl;
  NU_XYMN_1<MM, NN> nu;
#endif

// Velocity field in y direction
#if VEL == 2
  std::string filename("sqkt_bdry_2_");
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
