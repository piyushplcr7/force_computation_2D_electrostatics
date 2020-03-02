#ifndef FORCECALCULATIONHPP
#define FORCECALCULATIONHPP

#include <cassert>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "continuous_space.hpp"
#include "dirichlet.hpp"
#include "discontinuous_space.hpp"
#include "gauleg.hpp"
#include "integral_gauss.hpp"
#include "neumann.hpp"
#include "parametrized_mesh.hpp"
#include "single_layer.hpp"
#include <Eigen/Dense>

double sqrt_epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
double tol = std::numeric_limits<double>::epsilon();
double c = 0.5;

double xy_to_phi(double x, double y) {
  double phi;
  double thet;
  if (x == 0 && y == 0)
    return 0;
  thet = atan(abs(y / x));
  if (x > 0 && y == 0)
    return 0;
  else if (x > 0 && y > 0)
    return thet;
  else if (x == 0 && y > 0)
    return M_PI / 2;
  else if (x < 0 && y > 0)
    return M_PI - thet;
  else if (x < 0 && y == 0)
    return M_PI;
  else if (x < 0 && y < 0)
    return M_PI + thet;
  else if (x == 0 && y < 0)
    return 1.5 * M_PI;
  else if (x > 0 && y < 0)
    phi = 2 * M_PI - thet;
}

/*
 * This function solves the adjoint problem, which is required for calculating
 * the shape gradient. The adjoint problem is given as: \f$ a_V(p,\phi) =
 * -J(\phi) \f$. \f$ p \f$ is the adjoint solution and \f$ J \f$ is the
 * energy functional. This function uses the lowest order BEM space \f$ S^-1_0
 * \f$
 *
 * @param mesh The mesh of the domain over which the adjoint problem is to be
 * solved. The mesh has to be a ParametricMesh object.
 * @param g Function specifying the Dirichlet data
 * @param order Quadrature order to be used in numerical integration
 * @return An Eigen::VectorXd object containing the solution coefficients for
 * the adjoint solution
 */
Eigen::VectorXd SolveAdjoint(const parametricbem2d::ParametrizedMesh &mesh,
                             std::function<double(double, double)> g,
                             unsigned order) {
  // Lowest order trial and test BEM space
  parametricbem2d::DiscontinuousSpace<0> trial_space;
  // Getting the LHS of the adjoint equation
  Eigen::MatrixXd V =
      parametricbem2d::single_layer::GalerkinMatrix(mesh, trial_space, order);

  // Calculating the RHS vector in the adjoint equation

  // Number of reference shape functions for the space
  unsigned q = trial_space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = trial_space.getSpaceDim(numpanels);
  // Getting the panel vector
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Initializing the RHS vector with zeros
  Eigen::VectorXd rhs = Eigen::VectorXd::Constant(dims, 0);
  // Filling the RHS vector using local to global mapping
  for (unsigned i = 0; i < numpanels; ++i) {
    // Evaluating the local integral for a panel pi
    parametricbem2d::AbstractParametrizedCurve &gamma_pi = *panels[i];
    // Evaluating the local integral for the kth reference shape function
    for (unsigned k = 0; k < q; ++k) {
      // local integral arising from the energy shape functional
      auto integrand = [&](double s) {
        return trial_space.evaluateShapeFunction(k, s) *
               g(gamma_pi(s)(0), gamma_pi(s)(1)) *
               gamma_pi.Derivative(s).norm();
      };
      // Evaluating the local integral
      double local_integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      // Local to global map
      unsigned II = trial_space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      rhs(II) += local_integral;
    }
  }
  // Solving the linear system
  // Eigen::FullPivLU<Eigen::MatrixXd> dec(V);
  Eigen::HouseholderQR<Eigen::MatrixXd> dec(V);
  Eigen::VectorXd sol = dec.solve(-c * rhs);
  return sol;
}

class U_default {
public:
  double operator()(const Eigen::Vector2d &X) const { return 0; }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    return Eigen::Vector2d(0, 0);
  }
};

U_default udefault;

/*
 * This function is used to calculate the shape gradient of the energy
 * functional in the direction specified by the velocity field.
 *
 * @param mesh ParametricMesh object representing the domain at which the
 * shape gradient is to be evaluated
 * @tparam G template type for the dirichlet data. Should support evaluation
 * operator and a grad function for calculating the gradient.
 * @param g Function specifying the Dirichlet data
 * @tparam NU Template type for the velocity field. Should support evaluation
 * operator, grad function for calculating the gradient and dgrad1/dgrad2
 * functions for evaluating the second order derivatives.
 * @param nu Object of type NU, specifying the velocity field
 * @param order Quadrature order to be used in numerical integration
 * @param out std::ofstream type object for storing the output in a file.
 * @return The shape gradient value
 */
template <typename G, typename NU, typename U = U_default>
double CalculateForce(const parametricbem2d::ParametrizedMesh &mesh, const G &g,
                      const NU &nu, unsigned order, std::ofstream &out,
                      const U &u = udefault) {
  // Lambda function which evaluates the Dirichlet data using a function of the
  // form double (double, double)
  auto potential = [&](double x, double y) {
    Eigen::Vector2d point(x, y);
    return g(point);
  };
  // Getting the state solution using direct first kind BEM formulation
  Eigen::VectorXd state_sol =
      parametricbem2d::dirichlet_bvp::direct_first_kind::solve(mesh, potential,
                                                               order);
  // Getting the adjoint solution
  Eigen::VectorXd adj_sol = SolveAdjoint(mesh, potential, order);

  // Evaluating the BEM shape gradient formula
  double force = eval_BEM_sg(mesh, g, nu, state_sol, adj_sol, order);

  // Evaluating the boundary shape gradient formula
  class UU {
  public:
    double operator()(const Eigen::Vector2d &X) const {
      double x = X(0);
      double y = X(1);
      // double phi = atan(y / x);
      // double r = std::sqrt(x*x+y*y);
      // double phi = xy_to_phi(x,y);
      // return std::pow(r,2./3.)*sin(2./3.*phi);
      // return 0.5*(3+2*(x*x)/(ri*ri)-1+2*y/ri);
      // return log(R);
      // return log(R)*R/sqrt(x*x+y*y);
      // return 0;
      return log(sqrt(x * x + y * y));
      // return log(sqrt(x * x + y * y))+x*x+y*y-R*R;
      // return log(R);//+x*x+y*y-R*R;
      // return x+y;
      // return log(sqrt((x-0.5) * (x-0.5) + (y-0.5) * (y-0.5)));
      /*return 2.3774437510817346 +
             (y * (2. / (x * x + y * y) - 0.2222222222222222)) +
             (-1 + (2 * x * x) / (x * x + y * y)) *
                 (1.2 / (x * x + y * y) -
                  0.014814814814814815 * (x * x + y * y)) -
             2.164042561333445 * log(sqrt(x * x + y * y));*/
    }

    Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
      double x = X(0);
      double y = X(1);
      // double r = sqrt(x * x + y * y);
      double r = std::sqrt(x * x + y * y);
      // double phi = xy_to_phi(x,y);
      // return Eigen::Vector2d(0,0);
      // return Eigen::Vector2d(1, 1);
      // return log(R)*R*Eigen::Vector2d(-x/r/r/r,-y/r/r/r);
      //  return Eigen::Vector2d(2*x/ri/ri,1./ri);
      // return Eigen::Vector2d((x-0.5) / ((x-0.5) * (x-0.5) + (y-0.5) *
      // (y-0.5)),
      //                       (y-0.5) / ((x-0.5) * (x-0.5) + (y-0.5) *
      //                       (y-0.5)));
      return Eigen::Vector2d(x / (x * x + y * y), y / (x * x + y * y));
      // return Eigen::Vector2d(x / (x * x + y * y)+2*x, y / (x * x + y *
      // y)+2*y); return Eigen::Vector2d(2*x,2*y); return
      // Eigen::Vector2d(1+2*x,1+2*y);
      /*return Eigen::Vector2d((-0.02962962962962963*pow(x,11) +
      pow(x,9)*(-2.164042561333445 - 0.14814814814814814*pow(y,2)) +
      pow(x,5)*pow(y,2)*(2.3999999999999995 +
         y*(-12. - 12.984255368000667*y - 0.2962962962962963*pow(y,3))) +
      pow(x,7)*(-2.4 + y*(-4. - 8.65617024533378*y -
            0.2962962962962963*pow(y,3))) +
      pow(x,3)*pow(y,4)*(11.999999999999998 +
         y*(-12. - 8.65617024533378*y - 0.14814814814814814*pow(y,3))) +
      x*pow(y,6)*(7.199999999999999 +
         y*(-4. - 2.164042561333445*y - 0.02962962962962963*pow(y,3))))/
    pow(pow(x,2) + pow(y,2),5),
   (0. + pow(x,10)*(-0.2222222222222222 + 0.02962962962962963*y) +
      2.4*pow(y,7) - 2.*pow(y,8) - 2.164042561333445*pow(y,9) -
      0.2222222222222222*pow(y,10) + 0.02962962962962963*pow(y,11) +
      pow(x,8)*(2. + y*(-2.164042561333445 +
            (-1.111111111111111 + 0.14814814814814814*y)*y)) +
      pow(x,4)*pow(y,3)*(-11.999999999999998 +
         pow(y,2)*(-12.984255368000667 +
            (-2.222222222222222 + 0.2962962962962963*y)*y)) +
      pow(x,2)*pow(y,5)*(-2.3999999999999995 +
         y*(-4. + y*(-8.65617024533378 +
               (-1.111111111111111 + 0.14814814814814814*y)*y))) +
      pow(x,6)*y*(-7.199999999999999 +
         y*(4. + y*(-8.65617024533378 +
               (-2.222222222222222 + 0.2962962962962963*y)*y))))/
    pow(pow(x,2) + pow(y,2),5));*/
      // return 2./3.*std::pow(r,-1./3.)*Eigen::Vector2d(-sin(phi/3),
      // cos(phi/3));
    }
  };

  // U u;
  unsigned numpanels = mesh.getNumPanels();

  double bdry_sg_ex =
      numpanels < 30 ? eval_bdry_sg_ex(mesh, g, nu, state_sol, u, order) : 0;
  double bdry_sg1 = eval_bdry_sg1(mesh, g, nu, state_sol, order);
  double bdry_sg = eval_bdry_sg(mesh, g, nu, state_sol, order);

  // parametricbem2d::DiscontinuousSpace<0> discont;
  // parametricbem2d::ContinuousSpace<1> cont;
  // check potential
  /*auto checkpotential = [&](double x, double y) {
    return -c * potential(x, y);
  };
  Eigen::VectorXd check_adj_sol = SolveAdjoint(mesh, checkpotential, order);
  Eigen::Vector2d eval_pt;
  eval_pt << 2, 2;
  std::cout << "Adj_sol check: Evaluating potential at 2,0 : "
            << parametricbem2d::single_layer::Potential(
                   eval_pt, adj_sol, mesh, discont, order) << " , exact: " <<
  -c*4;*/

  // Eigen::MatrixXd M = parametricbem2d::MassMatrix(mesh,
  // discont,discont,order); std::cout << "Mass Matrix \n" << M << std::endl;
  // std::cout << "Second term from different procedure: " <<
  // state_sol.dot(M*adj_sol)*(-1./2./M_PI) << std::endl; Eigen::MatrixXd K =
  // parametricbem2d::double_layer::GalerkinMatrix(mesh,cont,discont,order);
  // Eigen::VectorXd con = Eigen::VectorXd::Constant(mesh.getNumPanels(),2);
  // std::cout << "FiveSix from different procedure: " << -(K *
  // con).dot(adj_sol) << std::endl;
  // std::cout << "mass vector routine: \n"
  //          << 2 * parametricbem2d::MassVector(mesh, discont, order)
  //          << std::endl;
  // Calculating the force the other way
  /*parametricbem2d::PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the space
  unsigned q = discont.getQ();
  unsigned qc = cont.getQ();
  // Getting the number of panels
  // Assumes inner and outer have same numpanels

  // Getting space dimensions
  unsigned dims = discont.getSpaceDim(numpanels);

  unsigned dimsc = cont.getSpaceDim(numpanels);

  // Initializing the matrix Q with zeros
  // for gradu.n gradu.n
  Eigen::MatrixXd Q = Eigen::MatrixXd::Constant(dims, dims, 0);

  // for gradu.t gradu.t
  Eigen::MatrixXd Qc = Eigen::MatrixXd::Constant(dimsc, dimsc, 0);
  // bintegral stores the exact value for c \int_{\Gamma} |gradu|^2 nu.n dS
  double bintegral = 0;

  double bintegralgu = 0;

  Eigen::VectorXd Q_gn_un = Eigen::VectorXd::Constant(dims, 0);
  Eigen::VectorXd Q_gt_ut_d = Eigen::VectorXd::Constant(dimsc, 0);

  Eigen::VectorXd g_N = cont.Interpolate(potential, mesh);

  // Looping over all the panels
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];

    if (numpanels < 30) {
      // boundary formula with exact knowledge of the function u(assumes u is
      // supplied as g) Use only to get the converged value fast
      auto bintegrand = [&](double s) {
        Eigen::Vector2d x = pi(s);
        Eigen::Vector2d tangent = pi.Derivative(s);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return g.grad(x).dot(g.grad(x)) * nu(x).dot(normal) *
               pi.Derivative(s).norm();
      };
      bintegral +=
          0.5 * parametricbem2d::ComputeIntegral(bintegrand, -1, 1, order);
    }

    if (numpanels < 30) {
      // boundary formula with exact knowledge of the function u and g
      auto bintegrandgu = [&](double s) {
        Eigen::Vector2d x = pi(s);
        Eigen::Vector2d tangent = pi.Derivative(s);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return (-0.5 * u.grad(x).dot(u.grad(x)) + g.grad(x).dot(u.grad(x))) *
               nu(x).dot(normal) * pi.Derivative(s).norm();
      };
      bintegralgu +=
          parametricbem2d::ComputeIntegral(bintegrandgu, -1, 1, order);
    }

    // Looping over the reference shape functions for (gradu.n)^2
    for (unsigned k = 0; k < q; ++k) {
      for (unsigned l = 0; l < q; ++l) {
        auto integrand = [&](double s) {
          Eigen::Vector2d tangent = pi.Derivative(s);
          Eigen::Vector2d normal;
          normal << tangent(1), -tangent(0);
          normal /= normal.norm();
          return discont.evaluateShapeFunction(k, s) *
                 discont.evaluateShapeFunction(l, s) * nu(pi(s)).dot(normal) *
                 pi.Derivative(s).norm();
        };
        double local_integral =
            parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
        // int II = discont.LocGlobMap(k + 1, i + 1, numpanels) - 1;
        // int JJ = discont.LocGlobMap(l + 1, i + 1, numpanels) - 1;
        int II = discont.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        int JJ = discont.LocGlobMap2(l + 1, i + 1, mesh) - 1;
        Q(II, JJ) += local_integral;
      }
    }

    // looping over rsf for cont space, evaluating integral for (gradu.t)^2
    for (unsigned k = 0; k < qc; ++k) {
      for (unsigned l = 0; l < qc; ++l) {
        auto integrandc = [&](double s) {
          Eigen::Vector2d tangent = pi.Derivative(s);
          Eigen::Vector2d normal;
          normal << tangent(1), -tangent(0);
          normal /= normal.norm();
          return cont.evaluateShapeFunctionDot(k, s) *
                 cont.evaluateShapeFunctionDot(l, s) * nu(pi(s)).dot(normal) /
                 pi.Derivative(s).norm();
        };
        double local_integralc =
            parametricbem2d::ComputeIntegral(integrandc, -1, 1, order);
        // int II = discont.LocGlobMap(k + 1, i + 1, numpanels) - 1;
        // int JJ = discont.LocGlobMap(l + 1, i + 1, numpanels) - 1;
        int II = cont.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        int JJ = cont.LocGlobMap2(l + 1, i + 1, mesh) - 1;
        Qc(II, JJ) += local_integralc;
      }
    }

    // looping over shape functions for (gradg.n)(gradu.n)
    for (unsigned k = 0; k < q; ++k) {
      auto integrand = [&](double s) {
        Eigen::Vector2d tangent = pi.Derivative(s);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return discont.evaluateShapeFunction(k, s) * g.grad(pi(s)).dot(normal) *
               nu(pi(s)).dot(normal) * pi.Derivative(s).norm();
      };
      double local_integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      // int II = discont.LocGlobMap(k + 1, i + 1, numpanels) - 1;
      // int JJ = discont.LocGlobMap(l + 1, i + 1, numpanels) - 1;
      int II = discont.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      Q_gn_un(II) += local_integral;
    }

    // looping over shape functions for (gradg.t)(gradu.t) discrete
    for (unsigned k = 0; k < qc; ++k) {
      auto integrand = [&](double s) {
        Eigen::Vector2d tangent = pi.Derivative(s);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return cont.evaluateShapeFunctionDot(k, s) *
               g.grad(pi(s)).dot(tangent) * nu(pi(s)).dot(normal) /
               pi.Derivative(s).norm();
      };
      double local_integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      // int II = discont.LocGlobMap(k + 1, i + 1, numpanels) - 1;
      // int JJ = discont.LocGlobMap(l + 1, i + 1, numpanels) - 1;
      int II = cont.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      Q_gt_ut_d(II) += local_integral;
    }
  }*/
  // Matrix Q evaluated

  /*double gradgn_gradun = Q_gn_un.dot(state_sol);
  double gradgt_gradut_d = Q_gt_ut_d.dot(g_N);
  double gradgt_gradut_e = 0;
  double gradut_gradut = g_N.dot(Qc * g_N);
  double gradun_gradun = state_sol.dot(Q * state_sol);*/

  // fother stores the integral of (gradu.n)^2
  // double fother = c * state_sol.dot(Q * state_sol);
  // stores the integral of (gradu.t)^2
  // double fotherpart = c * g_N.dot(Qc * g_N);
  // std::cout << "fotherpart " << fotherpart << std::endl;
  // std::cout << "Q matrix \n" << Q << std::endl;
  // Reduced state sol vector
  // std::cout << "dims: " << dims << std::endl;
  /*unsigned temp = dims / 2;
  // std::cout << "temp: " << temp << std::endl;
  Eigen::VectorXd a = state_sol.segment(0, temp);
  Eigen::VectorXd b = state_sol.segment(temp, temp);
  Eigen::MatrixXd Q1 = Q.block(0, 0, temp, temp);
  Eigen::MatrixXd Q2 = Q.block(temp, temp, temp, temp);*/

  // std::cout << "block 1 force : " << c*a.dot(Q1*a) << std::endl;
  // std::cout << "block 2 force : " << c*b.dot(Q2*b) << std::endl;
  std::cout.precision(std::numeric_limits<double>::digits10);
  std::cout << std::setw(10) << numpanels << std::setw(25) << 0 << std::setw(25)
            << force << std::setw(25) << 0 << std::setw(25) << bdry_sg1
            << std::setw(25) << bdry_sg << std::setw(25) << bdry_sg_ex
            << std::endl;
  // Writing the output
  out.precision(std::numeric_limits<double>::digits10);
  out << std::setw(10) << numpanels << std::setw(25) << 0 << std::setw(25)
      << force << std::setw(25) << 0 << std::setw(25) << bdry_sg1
      << std::setw(25) << bdry_sg << std::setw(25) << bdry_sg_ex << std::endl;
  return force;
}

/*
 * This function is used to calculate the BEM shape gradient formula of the
 * energy functional in the direction specified by the velocity field.
 *
 * @param mesh ParametricMesh object representing the domain at which the
 * shape gradient is to be evaluated
 * @tparam G template type for the dirichlet data. Should support evaluation
 * operator and a grad function for calculating the gradient.
 * @param g Function specifying the Dirichlet data
 * @tparam NU Template type for the velocity field. Should support evaluation
 * operator, grad function for calculating the gradient and dgrad1/dgrad2
 * functions for evaluating the second order derivatives.
 * @param nu Object of type NU, specifying the velocity field
 * @param order Quadrature order to be used in numerical integration
 * @param out std::ofstream type object for storing the output in a file.
 * @return The shape gradient value
 */
template <typename G, typename NU>
double eval_BEM_sg(const parametricbem2d::ParametrizedMesh &mesh, const G &g,
                   const NU &nu, const Eigen::VectorXd &state_sol,
                   const Eigen::VectorXd &adj_sol, unsigned order) {
  // Initializing the BEM shape gradient value
  double force = 0;
  // Evaluating the individual terms in the BEM shape gradient formula
  force += EvaluateFirst(mesh, g, nu, state_sol, order);
  force += EvaluateSecond(mesh, state_sol, adj_sol, nu, order);
  force += EvaluateThird(mesh, adj_sol, g, nu, order);
  force += EvaluateFourth(mesh, adj_sol, g, nu, order);
  force += EvaluateFiveSix(mesh, adj_sol, g, nu, order);
  force += EvaluateSeventh(mesh, g, nu, adj_sol, order);
  return force;
}

template <typename G, typename NU>
double eval_bdry_sg(const parametricbem2d::ParametrizedMesh &mesh, const G &g,
                    const NU &nu, const Eigen::VectorXd &state_sol,
                    unsigned order) {
  // Evaluating the boundary shape gradient formula
  parametricbem2d::DiscontinuousSpace<0> discont;
  parametricbem2d::ContinuousSpace<1> cont;
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the BEM spaces
  unsigned q = discont.getQ();
  unsigned qc = cont.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting dimensions of BEM spaces
  unsigned dims = discont.getSpaceDim(numpanels);
  unsigned dimsc = cont.getSpaceDim(numpanels);
  // Initializing matrix for evaluating gradu.n * gradu.n term
  Eigen::MatrixXd Q = Eigen::MatrixXd::Constant(dims, dims, 0);
  // Initializing matrix for evaluating gradu.t * gradu.t term
  Eigen::MatrixXd Qc = Eigen::MatrixXd::Constant(dimsc, dimsc, 0);
  // Initializing vector for evaluating gradg.n * gradu.n term
  Eigen::VectorXd Q_gn_un = Eigen::VectorXd::Constant(dims, 0);
  // Initializing vector for evaluating gradg.t * gradu.t term
  Eigen::VectorXd Q_gt_ut_d = Eigen::VectorXd::Constant(dimsc, 0);
  // Potential
  auto potential = [&](double x, double y) {
    Eigen::Vector2d point(x, y);
    return g(point);
  };
  // Interpolation of the Dirichlet data for the continuous BEM space
  Eigen::VectorXd g_N = cont.Interpolate(potential, mesh);
  // Looping over all the panels to evaluate the boundary shape gradient formula
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];

    // Looping over the reference shape functions for (gradu.n)^2
    for (unsigned k = 0; k < q; ++k) {
      for (unsigned l = 0; l < q; ++l) {
        // Local integrand
        auto integrand = [&](double s) {
          Eigen::Vector2d tangent = pi.Derivative(s);
          Eigen::Vector2d normal;
          normal << tangent(1), -tangent(0);
          normal /= normal.norm();
          return discont.evaluateShapeFunction(k, s) *
                 discont.evaluateShapeFunction(l, s) * nu(pi(s)).dot(normal) *
                 pi.Derivative(s).norm();
        };
        double local_integral =
            parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
        int II = discont.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        int JJ = discont.LocGlobMap2(l + 1, i + 1, mesh) - 1;
        Q(II, JJ) += local_integral;
      }
    }

    // looping over rsf for cont space, evaluating integral for (gradu.t)^2
    for (unsigned k = 0; k < qc; ++k) {
      for (unsigned l = 0; l < qc; ++l) {
        auto integrandc = [&](double s) {
          Eigen::Vector2d tangent = pi.Derivative(s);
          Eigen::Vector2d normal;
          normal << tangent(1), -tangent(0);
          normal /= normal.norm();
          return cont.evaluateShapeFunctionDot(k, s) *
                 cont.evaluateShapeFunctionDot(l, s) * nu(pi(s)).dot(normal) /
                 pi.Derivative(s).norm();
        };
        double local_integralc =
            parametricbem2d::ComputeIntegral(integrandc, -1, 1, order);
        int II = cont.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        int JJ = cont.LocGlobMap2(l + 1, i + 1, mesh) - 1;
        Qc(II, JJ) += local_integralc;
      }
    }

    // looping over shape functions for (gradg.n)(gradu.n)
    for (unsigned k = 0; k < q; ++k) {
      auto integrand = [&](double s) {
        Eigen::Vector2d tangent = pi.Derivative(s);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return discont.evaluateShapeFunction(k, s) * g.grad(pi(s)).dot(normal) *
               nu(pi(s)).dot(normal) * pi.Derivative(s).norm();
      };
      double local_integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      int II = discont.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      Q_gn_un(II) += local_integral;
    }

    // looping over shape functions for (gradg.t)(gradu.t) discrete
    for (unsigned k = 0; k < qc; ++k) {
      auto integrand = [&](double s) {
        Eigen::Vector2d tangent = pi.Derivative(s);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return cont.evaluateShapeFunctionDot(k, s) *
               g.grad(pi(s)).dot(tangent) * nu(pi(s)).dot(normal) /
               pi.Derivative(s).norm();
      };
      double local_integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      int II = cont.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      Q_gt_ut_d(II) += local_integral;
    }
  }
  double gradgn_gradun = Q_gn_un.dot(state_sol);
  double gradgt_gradut_d = Q_gt_ut_d.dot(g_N);
  double gradut_gradut = g_N.dot(Qc * g_N);
  double gradun_gradun = state_sol.dot(Q * state_sol);

  return c * gradgn_gradun - c * gradun_gradun +
         c * (gradgn_gradun + gradgt_gradut_d);
}

template <typename G, typename NU, typename U>
double eval_bdry_sg_ex(const parametricbem2d::ParametrizedMesh &mesh,
                       const G &g, const NU &nu,
                       const Eigen::VectorXd &state_sol, const U &u,
                       unsigned order) {
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Stores the boundary formula computed using knowledge of both g and u
  double bintegralgu = 0;
  // Looping over all the panels to evaluate the boundary shape gradient formula
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    // boundary formula with exact knowledge of the function u and g
    auto bintegrandgu = [&](double s) {
      Eigen::Vector2d x = pi(s);
      Eigen::Vector2d tangent = pi.Derivative(s);
      Eigen::Vector2d normal;
      normal << tangent(1), -tangent(0);
      normal /= normal.norm();
      return (-0.5 * u.grad(x).dot(u.grad(x)) + g.grad(x).dot(u.grad(x))) *
             nu(x).dot(normal) * pi.Derivative(s).norm();
    };
    bintegralgu += parametricbem2d::ComputeIntegral(bintegrandgu, -1, 1, order);
  }
  return bintegralgu;
}

template <typename G, typename NU>
double eval_bdry_sg1(const parametricbem2d::ParametrizedMesh &mesh, const G &g,
                     const NU &nu, const Eigen::VectorXd &state_sol,
                     unsigned order) {
  // Evaluating the boundary shape gradient formula
  parametricbem2d::DiscontinuousSpace<0> discont;
  parametricbem2d::ContinuousSpace<1> cont;
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the BEM spaces
  unsigned q = discont.getQ();
  unsigned qc = cont.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting dimensions of BEM spaces
  unsigned dims = discont.getSpaceDim(numpanels);
  unsigned dimsc = cont.getSpaceDim(numpanels);
  // Initializing matrix for evaluating gradu.n * gradu.n term
  Eigen::MatrixXd Q = Eigen::MatrixXd::Constant(dims, dims, 0);
  // Initializing matrix for evaluating gradu.t * gradu.t term
  Eigen::MatrixXd Qc = Eigen::MatrixXd::Constant(dimsc, dimsc, 0);
  // Initializing vector for evaluating gradg.n * gradu.n term
  Eigen::VectorXd Q_gn_un = Eigen::VectorXd::Constant(dims, 0);
  // Initializing vector for evaluating gradg.t * gradu.t term
  Eigen::VectorXd Q_gt_ut_d = Eigen::VectorXd::Constant(dimsc, 0);
  // Potential
  auto potential = [&](double x, double y) {
    Eigen::Vector2d point(x, y);
    return g(point);
  };
  // Interpolation of the Dirichlet data for the continuous BEM space
  Eigen::VectorXd g_N = cont.Interpolate(potential, mesh);
  // Looping over all the panels to evaluate the boundary shape gradient formula
  double gradut_gradut_e = 0;
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];

    // Looping over the reference shape functions for (gradu.n)^2
    for (unsigned k = 0; k < q; ++k) {
      for (unsigned l = 0; l < q; ++l) {
        // Local integrand
        auto integrand = [&](double s) {
          Eigen::Vector2d tangent = pi.Derivative(s);
          Eigen::Vector2d normal;
          normal << tangent(1), -tangent(0);
          normal /= normal.norm();
          return discont.evaluateShapeFunction(k, s) *
                 discont.evaluateShapeFunction(l, s) * nu(pi(s)).dot(normal) *
                 pi.Derivative(s).norm();
        };
        double local_integral =
            parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
        int II = discont.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        int JJ = discont.LocGlobMap2(l + 1, i + 1, mesh) - 1;
        Q(II, JJ) += local_integral;
      }
    }

    // looping over rsf for cont space, evaluating integral for (gradu.t)^2
    auto gradgt_gradgt = [&](double s) {
      Eigen::Vector2d tangent = pi.Derivative(s);
      Eigen::Vector2d normal;
      normal << tangent(1), -tangent(0);
      normal /= normal.norm();
      tangent /= tangent.norm();
      return std::pow(g.grad(pi(s)).dot(tangent), 2) * nu(pi(s)).dot(normal) *
             pi.Derivative(s).norm();
    };

    gradut_gradut_e +=
        parametricbem2d::ComputeIntegral(gradgt_gradgt, -1, 1, order);

    // looping over shape functions for (gradg.n)(gradu.n)
    for (unsigned k = 0; k < q; ++k) {
      auto integrand = [&](double s) {
        Eigen::Vector2d tangent = pi.Derivative(s);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return discont.evaluateShapeFunction(k, s) * g.grad(pi(s)).dot(normal) *
               nu(pi(s)).dot(normal) * pi.Derivative(s).norm();
      };
      double local_integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      int II = discont.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      Q_gn_un(II) += local_integral;
    }
  }
  double gradgn_gradun = Q_gn_un.dot(state_sol);
  double gradun_gradun = state_sol.dot(Q * state_sol);

  return c * gradgn_gradun - c * gradun_gradun +
         c * (gradgn_gradun + gradut_gradut_e);
}

/*
 * This function evaluates the first term in the 2D shape gradient formula
 *
 * @param mesh ParametricMesh object representing the domain for which the
 * computations have to be done
 * @tparam G template type for the dirichlet data. Should support evaluation
 * operator and a grad function for calculating the gradient.
 * @param g Function specifying the Dirichlet data
 * @tparam NU Template type for the velocity field. Should support evaluation
 * operator, grad function for calculating the gradient and dgrad1/dgrad2
 * functions for evaluating the second order derivatives.
 * @param nu Object of type NU, specifying the velocity field
 * @param state_sol Eigen::VectorXd type object storing the state solution
 * coefficients
 * @param order Quadrature order to be used in numerical integration
 * @return The first term in the 2D shape gradient formula
 */
template <typename G, typename NU>
double EvaluateFirst(const parametricbem2d::ParametrizedMesh &mesh, const G &g,
                     const NU &nu, const Eigen::VectorXd &state_sol,
                     unsigned order) {
  // The BEM space used for solving the state and adjoint problems
  parametricbem2d::DiscontinuousSpace<0> space;
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the space
  unsigned q = space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = space.getSpaceDim(numpanels);
  // Initializing the global vector V_{i}
  Eigen::VectorXd V = Eigen::VectorXd::Constant(dims, 0);
  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &gamma_pi = *panels[i];
    // Looping over the reference shape functions
    for (unsigned k = 0; k < q; ++k) {
      // Local integrand for the first term
      auto integrand = [&](double s) {
        return space.evaluateShapeFunction(k, s) *
               (g.grad(gamma_pi(s)).dot(nu(gamma_pi(s)))) *
               gamma_pi.Derivative(s).norm();
      };
      // Evaluating the local integral
      double integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      // Local to global mapping
      unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      // Adding the integral to the right global place
      V(II) += integral;
    }
  }
  // debugging with linear g and const nu -  vector V should give panel
  // lengths
  if (!true) {
    std::cout << "V vector \n" << V << std::endl;
  }
  // Evaluating the value of the first term
  return c * state_sol.dot(V);
}

/*
 * This function evaluates the seventh term in the 2D shape gradient formula
 *
 * @param mesh ParametricMesh object representing the domain for which the
 * computations have to be done
 * @tparam G template type for the dirichlet data. Should support evaluation
 * operator and a grad function for calculating the gradient.
 * @param g Function specifying the Dirichlet data
 * @tparam NU Template type for the velocity field. Should support evaluation
 * operator, grad function for calculating the gradient and dgrad1/dgrad2
 * functions for evaluating the second order derivatives.
 * @param nu Object of type NU, specifying the velocity field
 * @param adj_sol Eigen::VectorXd type object storing the adjoint solution
 * coefficients
 * @param order Quadrature order to be used in numerical integration
 * @return The seventh term in the 2D shape gradient formula
 */
template <typename G, typename NU>
double EvaluateSeventh(const parametricbem2d::ParametrizedMesh &mesh,
                       const G &g, const NU &nu, const Eigen::VectorXd &adj_sol,
                       unsigned order) {
  // The BEM space used for solving the state and adjoint problems
  parametricbem2d::DiscontinuousSpace<0> space;
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the space
  unsigned q = space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = space.getSpaceDim(numpanels);
  // Initializing the global vector V_{i}
  Eigen::VectorXd V = Eigen::VectorXd::Constant(dims, 0);
  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &gamma_pi = *panels[i];
    // Looping over the reference shape functions
    for (unsigned k = 0; k < q; ++k) {
      // Local integrand for the seventh term
      auto integrand = [&](double s) {
        return space.evaluateShapeFunction(k, s) *
               (g.grad(gamma_pi(s)).dot(nu(gamma_pi(s)))) *
               gamma_pi.Derivative(s).norm();
      };
      // Evaluating the local integral
      double integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      // Local to global mapping
      // unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
      unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      // Adding the integral to the right global place
      V(II) += integral;
    }
  }
  // debugging with linear g and const nu -  vector would give panel lengths
  if (!true) {
    std::cout << "V vector \n" << V << std::endl;
  }
  // Evaluating the value of the seventh term
  return -0.5 * adj_sol.dot(V);
}

/*
 * This function evaluates the second term in the 2D shape gradient formula
 *
 * @param mesh ParametricMesh object representing the domain for which the
 * computations have to be done
 * @tparam G template type for the dirichlet data. Should support evaluation
 * operator and a grad function for calculating the gradient.
 * @param g Function specifying the Dirichlet data
 * @tparam NU Template type for the velocity field. Should support evaluation
 * operator, grad function for calculating the gradient and dgrad1/dgrad2
 * functions for evaluating the second order derivatives.
 * @param nu Object of type NU, specifying the velocity field
 * @param state_sol Eigen::VectorXd type object storing the state solution
 * coefficients
 * @param adj_sol Eigen::VectorXd type object storing the adjoint solution
 * coefficients
 * @param order Quadrature order to be used in numerical integration
 * @return The second term in the 2D shape gradient formula
 */
template <typename NU>
double EvaluateSecond(const parametricbem2d::ParametrizedMesh &mesh,
                      const Eigen::VectorXd &state_sol,
                      const Eigen::VectorXd &adj_sol, const NU &nu,
                      unsigned order) {
  // The BEM space used for solving the state and adjoint problems
  parametricbem2d::DiscontinuousSpace<0> space;
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the space
  unsigned q = space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = space.getSpaceDim(numpanels);
  // Initializing the R_{ij} matrix
  Eigen::MatrixXd R = Eigen::MatrixXd::Constant(dims, dims, 0);

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    for (unsigned j = 0; j < numpanels; ++j) {
      // The panels pi and pi' for which the local integral has to be
      // evaluated
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {
        for (unsigned l = 0; l < q; ++l) {
          double local_integral = 0;
          // coinciding panels case
          if (i == j) {
            auto integrand = [&](double t, double s) {
              double non_singular = space.evaluateShapeFunction(k, t) *
                                    space.evaluateShapeFunction(l, s) *
                                    pi.Derivative(t).norm() *
                                    pi.Derivative(s).norm();
              double singular;
              // Direct evaluation when away from singularity
              if (fabs(s - t) > sqrt_epsilon) {

                singular = (pi(s) - pi(t)).dot(nu(pi(s)) - nu(pi(t))) /
                           (pi(s) - pi(t)).squaredNorm();

              }
              // stable evaluation near singularity using Taylor expansion
              else {
                singular = pi.Derivative((s + t) / 2.)
                               .dot(nu.grad(pi((s + t) / 2.)).transpose() *
                                    pi.Derivative((s + t) / 2.)) /
                           (pi.Derivative((s + t) / 2.)).squaredNorm();
              }

              return singular * non_singular;
            };
            // Function for upper limit of the integral
            auto ul = [&](double x) { return 1.; };
            // Function for lower limit of the integral
            auto ll = [&](double x) { return -1.; };
            local_integral = parametricbem2d::ComputeDoubleIntegral(
                integrand, -1., 1., ll, ul, order);
          }

          // Adjacent panels case
          else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
                   (pi(-1) - pi_p(1)).norm() / 100. < tol) {
            // Swap is used to check whether pi(1) = pi'(-1) or pi(-1) =
            // pi'(1)
            bool swap = (pi(1) - pi_p(-1)).norm() / 100. > tol;
            // Panel lengths for local arclength parametrization
            double length_pi =
                2 * pi.Derivative(swap ? -1 : 1)
                        .norm(); // Length for panel pi to ensure norm of
                                 // arclength parametrization is 1 at the
                                 // common point
            double length_pi_p =
                2 * pi_p.Derivative(swap ? 1 : -1)
                        .norm(); // Length for panel pi_p to ensure norm of
                                 // arclength parametrization is 1 at the
                                 // common point

            // Local integrand in polar coordinates
            auto integrand = [&](double phi, double r) {
              // Converting polar coordinates to local arclength coordinates
              double s_pr = r * cos(phi);
              double t_pr = r * sin(phi);
              // Converting local arclength coordinates to reference interval
              // coordinates
              double s = swap ? 1 - 2 * s_pr / length_pi_p
                              : 2 * s_pr / length_pi_p - 1;
              double t =
                  swap ? 2 * t_pr / length_pi - 1 : 1 - 2 * t_pr / length_pi;
              // reference interval coordinates corresponding to zeros in
              // arclength coordinates
              double s0 = swap ? 1 : -1;
              double t0 = swap ? -1 : 1;

              double non_singular =
                  space.evaluateShapeFunction(k, t) *
                  space.evaluateShapeFunction(l, s) * pi.Derivative(t).norm() *
                  pi_p.Derivative(s).norm() * (4 / length_pi / length_pi_p);
              double singular;
              // Direct evaluation away from the singularity
              if (r > sqrt_epsilon) {
                singular = (pi_p(s) - pi(t)).dot(nu(pi_p(s)) - nu(pi(t))) /
                           (pi_p(s) - pi(t)).squaredNorm();

              }
              // Stable evaluation near singularity using Taylor expansion
              else {
                singular =
                    (cos(phi) * pi_p.Derivative(s0) * 2 / length_pi_p +
                     sin(phi) * pi.Derivative(t0) * 2 / length_pi)
                        .dot(nu.grad(pi(t0)).transpose() *
                             (cos(phi) * pi_p.Derivative(s0) * 2 / length_pi_p +
                              sin(phi) * pi.Derivative(t0) * 2 / length_pi)) /
                    (1 + sin(2 * phi) *
                             pi.Derivative(t0).dot(pi_p.Derivative(s0)) * 4 /
                             length_pi / length_pi_p);
              }
              // Including the Jacobian of transformation 'r'
              return r * singular * non_singular;
            };
            // Getting the split point for integral over the angle in polar
            // coordinates
            double alpha = std::atan(length_pi / length_pi_p);
            // Defining upper and lower limits of inner integrals
            auto ll = [&](double phi) { return 0; };
            auto ul1 = [&](double phi) { return length_pi_p / cos(phi); };
            auto ul2 = [&](double phi) { return length_pi / sin(phi); };
            // Computing the local integral
            local_integral = parametricbem2d::ComputeDoubleIntegral(
                integrand, 0, alpha, ll, ul1, order);
            local_integral += parametricbem2d::ComputeDoubleIntegral(
                integrand, alpha, M_PI / 2., ll, ul2, order);
          }

          // General case
          else {
            // Local integral
            auto integrand = [&](double t, double s) {
              return space.evaluateShapeFunction(k, t) *
                     space.evaluateShapeFunction(l, s) *
                     pi.Derivative(t).norm() * pi_p.Derivative(s).norm() *
                     (pi_p(s) - pi(t)).dot(nu(pi_p(s)) - nu(pi(t))) /
                     (pi_p(s) - pi(t)).squaredNorm();
            };
            // Function for upper limit of the integral
            auto ul = [&](double x) { return 1.; };
            // Function for lower limit of the integral
            auto ll = [&](double x) { return -1.; };
            local_integral = parametricbem2d::ComputeDoubleIntegral(
                integrand, -1, 1, ll, ul, order);
          }

          // Local to global mapping
          unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
          unsigned JJ = space.LocGlobMap2(l + 1, j + 1, mesh) - 1;
          R(II, JJ) += local_integral;
        }
      }
    }
  }
  // with linear velocity, R matrix should contain the multiples of panel
  // lengths std::cout << "R matrix \n" << R << std::endl;
  return -1 / 2. / M_PI * state_sol.dot(R * adj_sol);
}

/*
 * This function evaluates the third term in the 2D shape gradient formula
 *
 * @param mesh ParametricMesh object representing the domain for which the
 * computations have to be done
 * @tparam G template type for the dirichlet data. Should support evaluation
 * operator and a grad function for calculating the gradient.
 * @param g Function specifying the Dirichlet data
 * @tparam NU Template type for the velocity field. Should support evaluation
 * operator, grad function for calculating the gradient and dgrad1/dgrad2
 * functions for evaluating the second order derivatives.
 * @param nu Object of type NU, specifying the velocity field
 * @param adj_sol Eigen::VectorXd type object storing the adjoint solution
 * coefficients
 * @param order Quadrature order to be used in numerical integration
 * @return The third term in the 2D shape gradient formula
 */
template <typename G, typename NU>
double EvaluateThird(const parametricbem2d::ParametrizedMesh &mesh,
                     const Eigen::VectorXd &adj_sol, const G &g, const NU &nu,
                     unsigned order) {
  // The BEM space used for solving the state and adjoint problems
  parametricbem2d::DiscontinuousSpace<0> space;
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the space
  unsigned q = space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = space.getSpaceDim(numpanels);
  // Initializing the S_{i} vector
  Eigen::VectorXd S = Eigen::VectorXd::Constant(dims, 0);
  double tol3 = 1e-14;

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    for (unsigned j = 0; j < numpanels; ++j) {
      // Local integrals evaluated for panels pi and pi'
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {
        double local_integral = 0;
        // coinciding panels case
        if (i == j) {
          auto integrand = [&](double t, double s) {
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(t).norm() *
                                  pi.Derivative(s).norm() * g(pi(t));
            // Getting the tangent vector to calculate the normal
            Eigen::Vector2d tangent = pi.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            // Stable evaluation away from singularity
            if (fabs(s - t) > sqrt_epsilon) {
              singular = (normal.dot(nu(pi(s)) - nu(pi(t))) -
                          (pi(s) - pi(t)).dot(nu.grad(pi(t)) * normal)) /
                         (pi(s) - pi(t)).squaredNorm();
            }
            // Stable evaluation near singularity using Taylor expansion
            else {
              Eigen::Vector2d vec(2); // zeta times gamma dot
              vec << (nu.dgrad1(pi(t)) * pi.Derivative(t))
                         .dot(pi.Derivative(t)),
                  (nu.dgrad2(pi(t)) * pi.Derivative(t)).dot(pi.Derivative(t));
              singular = 0.5 * normal.dot(vec) / pi.Derivative(t).squaredNorm();
            }
            return singular * non_singular;
          };
          // Function for upper limit of the integral
          auto ul = [&](double x) { return 1.; };
          // Function for lower limit of the integral
          auto ll = [&](double x) { return -1.; };
          local_integral = parametricbem2d::ComputeDoubleIntegral(
              integrand, -1, 1, ll, ul, order);
        }

        // Adjacent panels case
        else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
                 (pi(-1) - pi_p(1)).norm() / 100. < tol) {
          // Swap is used to check whether pi(1) = pi'(-1) or pi(-1) = pi'(1)
          bool swap = (pi(1) - pi_p(-1)).norm() / 100. >
                      std::numeric_limits<double>::epsilon();
          // Panel lengths for local arclength parametrization
          double length_pi =
              2 *
              pi.Derivative(swap ? -1 : 1)
                  .norm(); // Length for panel pi to ensure norm of arclength
                           // parametrization is 1 at the common point
          double length_pi_p =
              2 * pi_p.Derivative(swap ? 1 : -1)
                      .norm(); // Length for panel pi_p to ensure norm of
                               // arclength parametrization is 1 at the common
                               // point

          // Local integrand in polar coordinates
          auto integrand = [&](double phi, double r) {
            // Converting polar coordinates to local arclength coordinates
            double s_pr = r * cos(phi);
            double t_pr = r * sin(phi);
            // Converting local arclength coordinates to reference interval
            // coordinates
            double s =
                swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
            double t =
                swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;
            // Reference interval coordinates for zeros in arclength
            // coordinates
            double s0 = swap ? -1 : 1;
            double t0 = swap ? 1 : -1;
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(s).norm() * g(pi_p(t)) *
                                  pi_p.Derivative(t).norm() *
                                  (4 / length_pi / length_pi_p);
            // calculating the normal vector from the tangent vector
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            // Direct evaluation away from the singularity
            if (r > sqrt_epsilon) {
              singular = (normal.dot(nu(pi(s)) - nu(pi_p(t))) -
                          (pi(s) - pi_p(t)).dot(nu.grad(pi_p(t)) * normal)) /
                         (pi(s) - pi_p(t)).squaredNorm();
            }
            // Stable evaluation near singularity using Taylor expansion
            else {
              Eigen::Vector2d vec1;
              vec1 << pi.Derivative(s0).dot(nu.dgrad1(pi(s0)) *
                                            pi.Derivative(s0)),
                  pi.Derivative(s0).dot(nu.dgrad2(pi(s0)) * pi.Derivative(s0));
              Eigen::Vector2d vec2;
              vec2 << pi_p.Derivative(t0).dot(nu.dgrad1(pi_p(t0)) *
                                              pi_p.Derivative(t0)),
                  pi_p.Derivative(t0).dot(nu.dgrad2(pi_p(t0)) *
                                          pi_p.Derivative(t0));
              Eigen::Vector2d vec3;
              vec3 << pi_p.Derivative(t0).dot(nu.dgrad1(pi_p(t0)) *
                                              pi.Derivative(s0)),
                  pi_p.Derivative(t0).dot(nu.dgrad2(pi_p(t0)) *
                                          pi.Derivative(s0));
              singular = (normal.dot(2 * cos(phi) * cos(phi) * vec1 /
                                         length_pi / length_pi +
                                     2 * sin(phi) * sin(phi) * vec2 /
                                         length_pi_p / length_pi_p +
                                     4 * sin(phi) * cos(phi) * vec3 /
                                         length_pi / length_pi_p)) /
                         (1 + 4 * sin(2 * phi) *
                                  pi.Derivative(s0).dot(pi_p.Derivative(t0)) /
                                  length_pi / length_pi_p);
            }
            // including the jacobian r
            return r * singular * non_singular;
          };
          // Getting the split point for integral over the angle in polar
          // coordinates
          double alpha = std::atan(length_pi_p / length_pi);
          // Defining upper and lower limits of inner integrals
          auto ll = [&](double phi) { return 0; };
          auto ul1 = [&](double phi) { return length_pi / cos(phi); };
          auto ul2 = [&](double phi) { return length_pi_p / sin(phi); };
          local_integral = parametricbem2d::ComputeDoubleIntegral(
              integrand, 0, alpha, ll, ul1, order);
          local_integral += parametricbem2d::ComputeDoubleIntegral(
              integrand, alpha, M_PI / 2., ll, ul2, order);
        }

        // General case
        else {
          auto integrand = [&](double t, double s) {
            // Getting the normal vector from the tangent vector
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            return space.evaluateShapeFunction(k, s) * pi.Derivative(s).norm() *
                   pi_p.Derivative(t).norm() * g(pi_p(t)) *
                   (normal.dot(nu(pi(s)) - nu(pi_p(t))) -
                    (pi(s) - pi_p(t)).dot(nu.grad(pi_p(t)) * normal)) /
                   (pi(s) - pi_p(t)).squaredNorm();
          };
          // Function for upper limit of the integral
          auto ul = [&](double x) { return 1.; };
          // Function for lower limit of the integral
          auto ll = [&](double x) { return -1.; };
          local_integral = parametricbem2d::ComputeDoubleIntegral(
              integrand, -1, 1, ll, ul, order);
        }

        // Local to global mapping
        unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        S(II) += local_integral;
      }
    } // loop over pi_p ends
  }
  return -1. / 2. / M_PI * adj_sol.dot(S);
}

/*
 * This function evaluates the fourth term in the 2D shape gradient formula
 *
 * @param mesh ParametricMesh object representing the domain for which the
 * computations have to be done
 * @tparam G template type for the dirichlet data. Should support evaluation
 * operator and a grad function for calculating the gradient.
 * @param g Function specifying the Dirichlet data
 * @tparam NU Template type for the velocity field. Should support evaluation
 * operator, grad function for calculating the gradient and dgrad1/dgrad2
 * functions for evaluating the second order derivatives.
 * @param nu Object of type NU, specifying the velocity field
 * @param adj_sol Eigen::VectorXd type object storing the adjoint solution
 * coefficients
 * @param order Quadrature order to be used in numerical integration
 * @return The first term in the 2D shape gradient formula
 */
template <typename G, typename NU>
double EvaluateFourth(const parametricbem2d::ParametrizedMesh &mesh,
                      const Eigen::VectorXd &adj_sol, const G &g, const NU &nu,
                      unsigned order) {
  // The BEM space used for solving the state and adjoint problems
  parametricbem2d::DiscontinuousSpace<0> space;
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the space
  unsigned q = space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = space.getSpaceDim(numpanels);
  // Initializing the T_{i} vector
  Eigen::VectorXd T = Eigen::VectorXd::Constant(dims, 0);
  Eigen::MatrixXd K(dims, dims);

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    for (unsigned j = 0; j < numpanels; ++j) {
      // Local integral evaluated for panel pi and pi'
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {
        double local_integral = 0;
        // coinciding panels case
        if (i == j) {
          auto integrand = [&](double t, double s) {
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(t).norm() *
                                  pi.Derivative(s).norm() * g(pi(t));
            // Getting the normal vector from the tangent vector
            Eigen::Vector2d tangent = pi.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            // Direct evaluation when away from singularity
            if (fabs(s - t) > sqrt_epsilon) {
              singular = ((pi(s) - pi(t)).dot(normal) *
                          (pi(s) - pi(t)).dot(nu(pi(s)) - nu(pi(t)))) /
                         (pi(s) - pi(t)).squaredNorm() /
                         (pi(s) - pi(t)).squaredNorm();

            }
            // stable evaluation near singularity using Taylor expansion
            else {
              singular = 0.5 * pi.DoubleDerivative(t).dot(normal) *
                         pi.Derivative(t).dot(nu.grad(pi(t)).transpose() *
                                              pi.Derivative(t)) /
                         pi.Derivative(t).squaredNorm() /
                         pi.Derivative(t).squaredNorm();
            }

            return singular * non_singular;
          };
          // Function for upper limit of the integral
          auto ul = [&](double x) { return 1.; };
          // Function for lower limit of the integral
          auto ll = [&](double x) { return -1.; };
          local_integral = parametricbem2d::ComputeDoubleIntegral(
              integrand, -1., 1., ll, ul, order);
        }

        // Adjacent panels case
        else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
                 (pi(-1) - pi_p(1)).norm() / 100. < tol) {
          // Swap is used to check whether pi(1) = pi'(-1) or pi(-1) = pi'(1)
          bool swap = (pi(1) - pi_p(-1)).norm() / 100. >
                      std::numeric_limits<double>::epsilon();
          // Panel lengths for local arclength parametrization
          double length_pi =
              2 *
              pi.Derivative(swap ? -1 : 1)
                  .norm(); // Length for panel pi to ensure norm of arclength
                           // parametrization is 1 at the common point
          double length_pi_p =
              2 * pi_p.Derivative(swap ? 1 : -1)
                      .norm(); // Length for panel pi_p to ensure norm of
                               // arclength parametrization is 1 at the common
                               // point

          // Local integrand in polar coordinates
          auto integrand = [&](double phi, double r) {
            // Converting polar coordinates to local arclength coordinates
            double s_pr = r * cos(phi);
            double t_pr = r * sin(phi);
            // Converting local arclength coordinates to reference interval
            // coordinates
            double s =
                swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
            double t =
                swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;
            // Reference interval coordinates for zeros in arclength
            // coordinates
            double s0 = swap ? -1 : 1;
            double t0 = swap ? 1 : -1;
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(s).norm() * g(pi_p(t)) *
                                  pi_p.Derivative(t).norm() *
                                  (4 / length_pi / length_pi_p);
            // normal vector using the tangent vector
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            // Direct evaluation away from the singularity
            if (r > sqrt_epsilon) {
              singular = r *
                         ((pi(s) - pi_p(t)).dot(normal) *
                          (pi(s) - pi_p(t)).dot(nu(pi(s)) - nu(pi_p(t)))) /
                         (pi(s) - pi_p(t)).squaredNorm() /
                         (pi(s) - pi_p(t)).squaredNorm();

            }
            // Stable evaluation near singularity using Taylor expansion
            else {
              Eigen::VectorXd vec =
                  2 * cos(phi) * pi.Derivative(s0) / length_pi +
                  2 * sin(phi) * pi_p.Derivative(t0) / length_pi_p;
              vec *= swap ? 1 : -1;

              singular = vec.dot(normal) *
                         (vec.dot(nu.grad(pi(s0)).transpose() * vec)) /
                         (1 + 4 * sin(2 * phi) *
                                  pi.Derivative(s0).dot(pi_p.Derivative(t0)) /
                                  length_pi / length_pi_p) /
                         (1 + 4 * sin(2 * phi) *
                                  pi.Derivative(s0).dot(pi_p.Derivative(t0)) /
                                  length_pi / length_pi_p);
            }
            return singular * non_singular;
          };
          // Getting the split point for integral over the angle in polar
          // coordinates
          double alpha = std::atan(length_pi_p / length_pi);
          // Defining upper and lower limits of inner integrals
          auto ll = [&](double phi) { return 0; };
          auto ul1 = [&](double phi) { return length_pi / cos(phi); };
          auto ul2 = [&](double phi) { return length_pi_p / sin(phi); };
          local_integral = parametricbem2d::ComputeDoubleIntegral(
              integrand, 0, alpha, ll, ul1, order);
          // std::cout << "evaluated local integral = " << local_integral <<
          // std::endl;
          local_integral += parametricbem2d::ComputeDoubleIntegral(
              integrand, alpha, M_PI / 2., ll, ul2, order);
        }

        // General case
        else {
          auto integrand = [&](double t, double s) {
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            return space.evaluateShapeFunction(k, s) * pi.Derivative(s).norm() *
                   pi_p.Derivative(t).norm() * g(pi_p(t)) *
                   (pi(s) - pi_p(t)).dot(normal) *
                   (pi(s) - pi_p(t)).dot(nu(pi(s)) - nu(pi_p(t))) /
                   (pi(s) - pi_p(t)).squaredNorm() /
                   (pi(s) - pi_p(t)).squaredNorm();
          };
          // Function for upper limit of the integral
          auto ul = [&](double x) { return 1.; };
          // Function for lower limit of the integral
          auto ll = [&](double x) { return -1.; };
          local_integral = parametricbem2d::ComputeDoubleIntegral(
              integrand, -1, 1, ll, ul, order);
        }

        // Local to global mapping
        unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        T(II) += local_integral;
      }
    } // loop over pi_p ends
  }
  // T vector for linear velocity field is simply 2pi * K * g_N
  if (!true) {
    auto potential = [&](double x, double y) {
      Eigen::Vector2d point(x, y);
      return g(point);
    };
    parametricbem2d::DiscontinuousSpace<0> discont;
    parametricbem2d::ContinuousSpace<1> cont;
    Eigen::MatrixXd K = parametricbem2d::double_layer::GalerkinMatrix(
        mesh, cont, discont, order);
    Eigen::VectorXd g_N = cont.Interpolate(potential, mesh);
    // std::cout << "4th term debugging: " << (2*M_PI * K *
    // g_N-T).norm()/T.norm() << std::endl; std::cout << "2pkg\n" << 2*M_PI *
    // K
    // * g_N << std::endl;
    std::cout << "t vector\n" << T << std::endl;
  }
  return 1. / M_PI * adj_sol.dot(T);
}

/*
 * This function evaluates the fifth and sixth terms in the 2D shape gradient
 * formula
 *
 * @param mesh ParametricMesh object representing the domain for which the
 * computations have to be done
 * @tparam G template type for the dirichlet data. Should support evaluation
 * operator and a grad function for calculating the gradient.
 * @param g Function specifying the Dirichlet data
 * @tparam NU Template type for the velocity field. Should support evaluation
 * operator, grad function for calculating the gradient and dgrad1/dgrad2
 * functions for evaluating the second order derivatives.
 * @param nu Object of type NU, specifying the velocity field
 * @param adj_sol Eigen::VectorXd type object storing the adjoint solution
 * coefficients
 * @param order Quadrature order to be used in numerical integration
 * @return The fifth and sixth terms in the 2D shape gradient formula combined
 */
template <typename G, typename NU>
double EvaluateFiveSix(const parametricbem2d::ParametrizedMesh &mesh,
                       const Eigen::VectorXd &adj_sol, const G &g, const NU &nu,
                       unsigned order) {
  // The BEM space used for solving the state and adjoint problems
  parametricbem2d::DiscontinuousSpace<0> space;
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the space
  unsigned q = space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = space.getSpaceDim(numpanels);
  // Initializing the U_{i} vector
  Eigen::VectorXd U = Eigen::VectorXd::Constant(dims, 0);

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    for (unsigned j = 0; j < numpanels; ++j) {
      // The panels pi and pi' for which the local integral has to be
      // evaluated
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {
        double local_integral = 0;
        // coinciding panels case
        if (i == j) {
          auto integrand = [&](double t, double s) {
            double non_singular =
                space.evaluateShapeFunction(k, s) * pi.Derivative(t).norm() *
                pi.Derivative(s).norm() *
                (g(pi(t)) * nu.div(pi(t)) + g.grad(pi(t)).dot(nu(pi(t))));
            // Calculating the normal vector using the tangent vector
            Eigen::Vector2d tangent = pi.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            // Direct evaluation when away from singularity
            if (fabs(s - t) > sqrt_epsilon)
              singular =
                  (pi(s) - pi(t)).dot(normal) / (pi(s) - pi(t)).squaredNorm();
            // stable evaluation near singularity using Taylor expansion
            else
              singular = 0.5 * pi.DoubleDerivative(t).dot(normal) /
                         pi.Derivative(t).squaredNorm();
            return singular * non_singular;
          };
          // Function for upper limit of the integral
          auto ul = [&](double x) { return 1.; };
          // Function for lower limit of the integral
          auto ll = [&](double x) { return -1.; };
          local_integral = parametricbem2d::ComputeDoubleIntegral(
              integrand, -1., 1., ll, ul, order);
        }

        // Adjacent panels case
        else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
                 (pi(-1) - pi_p(1)).norm() / 100. < tol) {
          // Swap is used to check whether pi(1) = pi'(-1) or pi(-1) = pi'(1)
          bool swap = (pi(1) - pi_p(-1)).norm() / 100. >
                      std::numeric_limits<double>::epsilon();
          // Panel lengths for local arclength parametrization
          double length_pi =
              2 *
              pi.Derivative(swap ? -1 : 1)
                  .norm(); // Length for panel pi to ensure norm of arclength
                           // parametrization is 1 at the common point
          double length_pi_p =
              2 * pi_p.Derivative(swap ? 1 : -1)
                      .norm(); // Length for panel pi_p to ensure norm of
                               // arclength parametrization is 1 at the common
                               // point

          // Local integrand in polar coordinates
          auto integrand = [&](double phi, double r) {
            // Converting polar coordinates to local arclength coordinates
            double s_pr = r * cos(phi);
            double t_pr = r * sin(phi);
            // Converting local arclength coordinates to reference interval
            // coordinates
            double s =
                swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
            double t =
                swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;

            double non_singular =
                space.evaluateShapeFunction(k, s) * pi.Derivative(s).norm() *
                pi_p.Derivative(t).norm() * (4 / length_pi / length_pi_p) *
                (g(pi_p(t)) * nu.div(pi_p(t)) +
                 g.grad(pi_p(t)).dot(nu(pi_p(t))));
            // normal vector
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            // Direct evaluation away from the singularity
            if (r > sqrt_epsilon) {
              singular = r * (pi(s) - pi_p(t)).dot(normal) /
                         (pi(s) - pi_p(t)).squaredNorm();
            }
            // Stable evaluation near singularity using Taylor expansion
            else {
              Eigen::VectorXd vec =
                  2 * cos(phi) * pi.Derivative(s) / length_pi +
                  2 * sin(phi) * pi_p.Derivative(t) / length_pi_p;
              vec *= swap ? 1 : -1;

              singular = vec.dot(normal) /
                         (1 + 4 * sin(2 * phi) *
                                  pi.Derivative(s).dot(pi_p.Derivative(t)) /
                                  length_pi / length_pi_p);
            }
            return singular * non_singular;
          };
          // Getting the split point for integral over the angle in polar
          // coordinates
          double alpha = std::atan(length_pi_p / length_pi);
          // Defining upper and lower limits of inner integrals
          auto ll = [&](double phi) { return 0; };
          auto ul1 = [&](double phi) { return length_pi / cos(phi); };
          auto ul2 = [&](double phi) { return length_pi_p / sin(phi); };
          local_integral = parametricbem2d::ComputeDoubleIntegral(
              integrand, 0, alpha, ll, ul1, order);
          local_integral += parametricbem2d::ComputeDoubleIntegral(
              integrand, alpha, M_PI / 2., ll, ul2, order);
        }

        // General case
        else {
          auto integrand = [&](double t, double s) {
            // normal vector
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            return space.evaluateShapeFunction(k, s) * pi.Derivative(s).norm() *
                   pi_p.Derivative(t).norm() *
                   (g(pi_p(t)) * nu.div(pi_p(t)) +
                    g.grad(pi_p(t)).dot(nu(pi_p(t)))) *
                   (pi(s) - pi_p(t)).dot(normal) /
                   (pi(s) - pi_p(t)).squaredNorm();
          };
          // Function for upper limit of the integral
          auto ul = [&](double x) { return 1.; };
          // Function for lower limit of the integral
          auto ll = [&](double x) { return -1.; };
          local_integral = parametricbem2d::ComputeDoubleIntegral(
              integrand, -1, 1, ll, ul, order);
        }

        // Local to global mapping
        unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        U(II) += local_integral;
      }
    } // loop over pi_p ends
  }
  // U vector for linear velocity field and constant g is double layer for
  // const function
  if (!true) {
    auto potential = [&](double x, double y) {
      Eigen::Vector2d point(x, y);
      return g(point) * nu.div(point) + g.grad(point).dot(nu(point));
    };
    parametricbem2d::DiscontinuousSpace<0> discont;
    parametricbem2d::ContinuousSpace<1> cont;
    Eigen::MatrixXd K = parametricbem2d::double_layer::GalerkinMatrix(
        mesh, cont, discont, order);
    Eigen::VectorXd g_N = cont.Interpolate(potential, mesh);
    // std::cout << "4th term debugging: " << (2*M_PI * K *
    // g_N-T).norm()/T.norm() << std::endl;
    std::cout << "2pkg\n" << 2 * M_PI * K * g_N << std::endl;
    std::cout << "u vector\n" << U << std::endl;
  }
  return -1. / 2. / M_PI * adj_sol.dot(U);
}

#endif
