#ifndef FORCECALCULATIONHPP
#define FORCECALCULATIONHPP

#include <cassert>
#include <cmath>
#include <exception>
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

Eigen::VectorXd SolveAdjoint(const parametricbem2d::ParametrizedMesh &mesh,
                             std::function<double(double, double)> g,
                             unsigned order) {
  // Trial and test space
  parametricbem2d::DiscontinuousSpace<0> trial_space;
  // Getting the LHS
  Eigen::MatrixXd V =
      parametricbem2d::single_layer::GalerkinMatrix(mesh, trial_space, order);
  // Calculating the rhs vector
  // Number of reference shape functions for the space
  unsigned q = trial_space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = trial_space.getSpaceDim(numpanels);
  // Getting the panel vector
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Initializing the rhs vector
  Eigen::VectorXd rhs = Eigen::VectorXd::Constant(dims, 0);
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &gamma_pi = *panels[i];
    for (unsigned k = 0; k < q; ++k) {
      auto integrand = [&](double s) {
        return trial_space.evaluateShapeFunction(k, s) *
               g(gamma_pi(s)(0), gamma_pi(s)(1)) *
               gamma_pi.Derivative(s).norm();
      };
      // Evaluating the local integral
      double local_integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      // Local to global map
      unsigned II = trial_space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
      rhs(II) += local_integral;
    }
  }
  Eigen::FullPivLU<Eigen::MatrixXd> dec(V);
  Eigen::VectorXd sol = dec.solve(-c * rhs);
  return sol;
}

template <typename G, typename NU>
double CalculateForce(const parametricbem2d::ParametrizedMesh &mesh_f, const G &g,
                      const NU &nu, unsigned order) {
  double force = 0;
  auto potential = [&](double x, double y) {
    Eigen::Vector2d point(x, y);
    return g(point);
  };
  Eigen::VectorXd state_sol_f =
      parametricbem2d::dirichlet_bvp::direct_first_kind::solve(mesh_f, potential,
                                                               order);
  // std::cout << "Evaluated the state solution! \n" << state_sol << std::endl;
  Eigen::VectorXd adj_sol_f = SolveAdjoint(mesh_f, potential, order);

  unsigned numpanels = mesh_f.getNumPanels();
  unsigned temp = numpanels/2;
  Eigen::VectorXd state_sol = state_sol_f.segment(0,temp);
  Eigen::VectorXd adj_sol = adj_sol_f.segment(0,temp);
  parametricbem2d::PanelVector panels_f = mesh_f.getPanels();
  parametricbem2d::PanelVector panels;
  panels.insert(panels.end(),panels_f.begin(),panels_f.begin()+temp);
  parametricbem2d::ParametrizedMesh mesh(panels);

  // std::cout << "adj sool \n" << adj_sol << std::endl;
  // std::cout << "Evaluated the adjoint solution! \n" << adj_sol << std::endl;
  double first = EvaluateFirst(mesh, g, nu, state_sol, order);
  // std::cout << "Evaluated the term 1! " << first << std::endl;
  double second = EvaluateSecond(mesh, state_sol, adj_sol, nu, order);
  // std::cout << "Evaluated the term 2! " << second << std::endl;
  double third = EvaluateThird(mesh, adj_sol, g, nu, order);
  // std::cout << "Evaluated the term 3!" << third << std::endl;
  double fourth = EvaluateFourth(mesh, adj_sol, g, nu, order);
  // std::cout << "Evaluated the term 4!" << fourth << std::endl;
  double fivesix = EvaluateFiveSix(mesh, adj_sol, g, nu, order);
  // std::cout << "Evaluated the term 5,6!" << fivesix << std::endl;
  double seven = EvaluateSeventh(mesh, g, nu, adj_sol, order);
  // std::cout << "Evaluated the term 7!" << seven << std::endl;
  force += first;
  force += second;
  force += third;
  force += fourth;
  force += fivesix;
  force += seven;
  // Check for fourth term for linear velocity
  if (true) {
    parametricbem2d::DiscontinuousSpace<0> discont;
    parametricbem2d::ContinuousSpace<1> cont;
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
    parametricbem2d::PanelVector panels = mesh.getPanels();
    // Number of reference shape functions for the space
    unsigned q = discont.getQ();
    // Getting the number of panels
    // Assumes inner and outer have same numpanels
    unsigned numpanels = mesh.getNumPanels();
    // Getting space dimensions
    unsigned dims = discont.getSpaceDim(numpanels);

    // Initializing the matrix Q with zeros
    Eigen::MatrixXd Q = Eigen::MatrixXd::Constant(dims, dims, 0);
    // Looping over all the panels
    for (unsigned i = 0; i < numpanels; ++i) {
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      // Looping over the reference shape functions
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
          int II = discont.LocGlobMap(k + 1, i + 1, numpanels) - 1;
          int JJ = discont.LocGlobMap(l + 1, i + 1, numpanels) - 1;
          Q(II, JJ) += local_integral;
        }
      }
    }
    // Matrix Q evaluated
    double fother = c * state_sol.dot(Q * state_sol);
    //std::cout << "Q matrix \n" << Q << std::endl;
    // Reduced state sol vector
    //std::cout << "dims: " << dims << std::endl;
    unsigned temp = dims/2;
    //std::cout << "temp: " << temp << std::endl;
    Eigen::VectorXd a = state_sol.segment(0,temp);
    Eigen::VectorXd b = state_sol.segment(temp,temp);
    Eigen::MatrixXd Q1 = Q.block(0,0,temp,temp);
    Eigen::MatrixXd Q2 = Q.block(temp,temp,temp,temp);
    //std::cout << "block 1 force : " << c*a.dot(Q1*a) << std::endl;
    //std::cout << "block 2 force : " << c*b.dot(Q2*b) << std::endl;
    std::cout << std::setw(10) << numpanels << std::setw(15) << fother << std::setw(15) << force << std::endl;
    //std::cout << std::setw(15) << fother ;
  }
  return force;
}

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
      // Integrand
      auto integrand = [&](double s) {
        return space.evaluateShapeFunction(k, s) *
               (g.grad(gamma_pi(s)).dot(nu(gamma_pi(s)))) *
               gamma_pi.Derivative(s).norm();
      };
      // Evaluating the integral
      double integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      // Local to global mapping
      unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
      // Adding the integral to the right global place
      V(II) += integral;
    }
  }
  // std::cout << "V vector \n" << V << std::endl;
  // Evaluating the first term by taking dot product
  return c * state_sol.dot(V);
}

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
      // Integrand
      auto integrand = [&](double s) {
        return space.evaluateShapeFunction(k, s) *
               (g.grad(gamma_pi(s)).dot(nu(gamma_pi(s)))) *
               gamma_pi.Derivative(s).norm();
      };
      // Evaluating the integral
      double integral =
          parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
      // Local to global mapping
      unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
      // Adding the integral to the right global place
      V(II) += integral;
    }
  }
  // Evaluating the first term by taking dot product
  return -0.5 * adj_sol.dot(V);
}

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
  bool linear = !true;

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    for (unsigned j = 0; j < numpanels; ++j) {

      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {
        for (unsigned l = 0; l < q; ++l) {
          double local_integral = 0;
          // coinciding panels
          if (i == j) {
            auto integrand = [&](double t, double s) {
              double non_singular = space.evaluateShapeFunction(k, t) *
                                    space.evaluateShapeFunction(l, s) *
                                    pi.Derivative(t).norm() *
                                    pi.Derivative(s).norm();
              double singular;
              if (fabs(s - t) > sqrt_epsilon) {
                if (!linear) {
                  singular = (pi(s) - pi(t)).dot(nu(pi(s)) - nu(pi(t))) /
                             (pi(s) - pi(t)).squaredNorm();
                } else {
                  singular = 1; //(pi(s) - pi_p(t)).dot(normal) / (pi(s) -
                                // pi_p(t)).squaredNorm();;
                }
              }

              else {
                if (!linear) {
                  singular = pi.Derivative((s + t) / 2.)
                                 .dot(nu.grad(pi((s + t) / 2.)).transpose() *
                                      pi.Derivative((s + t) / 2.)) /
                             (pi.Derivative((s + t) / 2.)).squaredNorm();
                } else {
                  singular =
                      1; // 0.5 * pi.DoubleDerivative(0.5 * (t + s)).dot(normal)
                         // / pi.Derivative(0.5 * (t + s)).squaredNorm();
                }
              }

              return singular * non_singular;
            };
            // Function for upper limit of the integral
            auto ul = [&](double x) { return 1.; };
            // Function for lower limit of the integral
            auto ll = [&](double x) { return -1.; };
            local_integral =
                ComputeDoubleIntegral(integrand, -1., 1., ll, ul, order);
          }

          // Adjacent
          else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
                   (pi(-1) - pi_p(1)).norm() / 100. < tol) {
            bool swap = (pi(1) - pi_p(-1)).norm() / 100. > tol;
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

            auto integrand = [&](double phi, double r) {
              // Local arclength coordinates
              double s_pr = r * cos(phi);
              double t_pr = r * sin(phi);
              // Reference interval coordinates
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
              if (r > sqrt_epsilon) {
                if (!linear) {
                  singular = (pi_p(s) - pi(t)).dot(nu(pi_p(s)) - nu(pi(t))) /
                             (pi_p(s) - pi(t)).squaredNorm();
                } else {
                  singular = 1; // r * (pi(s) - pi_p(t)).dot(normal) /
                                //(pi(s) - pi_p(t)).squaredNorm();
                }

              } else {
                if (!linear) {
                  singular =
                      (cos(phi) * pi_p.Derivative(s0) * 2 / length_pi_p +
                       sin(phi) * pi.Derivative(t0) * 2 / length_pi)
                          .dot(nu.grad(pi(t0)).transpose() *
                               (cos(phi) * pi_p.Derivative(s0) * 2 /
                                    length_pi_p +
                                sin(phi) * pi.Derivative(t0) * 2 / length_pi)) /
                      (1 + sin(2 * phi) *
                               pi.Derivative(t0).dot(pi_p.Derivative(s0)) * 4 /
                               length_pi / length_pi_p);
                } else {
                  Eigen::Vector2d b_r_phi =
                      (swap ? 1 : -1) *
                      (pi.Derivative(s0) * cos(phi) * 2 / length_pi +
                       pi_p.Derivative(t0) * sin(phi) * 2 / length_pi_p);
                  singular = 1; // b_r_phi.dot(normal) / b_r_phi.squaredNorm();
                }
              }
              // Including the jacobian r
              if (!linear)
                return r * singular * non_singular;
              else
                return singular * non_singular;
            };
            // Getting the split point for integral
            double alpha = std::atan(length_pi / length_pi_p);
            // Defining upper and lower limits of inner integrals
            auto ll = [&](double phi) { return 0; };
            auto ul1 = [&](double phi) { return length_pi_p / cos(phi); };
            auto ul2 = [&](double phi) { return length_pi / sin(phi); };
            local_integral =
                ComputeDoubleIntegral(integrand, 0, alpha, ll, ul1, order);
            local_integral += ComputeDoubleIntegral(integrand, alpha, M_PI / 2.,
                                                    ll, ul2, order);
          }

          // General
          else {
            auto integrand = [&](double t, double s) {
              if (!linear) {
                return space.evaluateShapeFunction(k, t) *
                       space.evaluateShapeFunction(l, s) *
                       pi.Derivative(t).norm() * pi_p.Derivative(s).norm() *
                       (pi_p(s) - pi(t)).dot(nu(pi_p(s)) - nu(pi(t))) /
                       (pi_p(s) - pi(t)).squaredNorm();
              } else {
                return space.evaluateShapeFunction(k, t) *
                       space.evaluateShapeFunction(l, s) *
                       pi.Derivative(t).norm() * pi_p.Derivative(s).norm();
              }
            };
            // Function for upper limit of the integral
            auto ul = [&](double x) { return 1.; };
            // Function for lower limit of the integral
            auto ll = [&](double x) { return -1.; };
            local_integral =
                ComputeDoubleIntegral(integrand, -1, 1, ll, ul, order);
          }

          // Local to global mapping
          unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
          unsigned JJ = space.LocGlobMap(l + 1, j + 1, numpanels) - 1;
          R(II, JJ) += local_integral;
        }
      }
    }
  }
  // std::cout << "R matrix \n" << R << std::endl;
  return -1 / 2. / M_PI * state_sol.dot(R * adj_sol);
}

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
    // evaluating local integrals for all pi_p, with fixed pi and summing
    for (unsigned j = 0; j < numpanels; ++j) {

      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {
        double local_integral = 0;
        // coinciding panels
        if (i == j) {
          auto integrand = [&](double t, double s) {
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(t).norm() *
                                  pi.Derivative(s).norm() * g(pi(t));

            Eigen::Vector2d tangent = pi.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            if (fabs(s - t) > sqrt_epsilon)
              // if (fabs(s - t) > tol3)
              singular = (normal.dot(nu(pi(s)) - nu(pi(t))) -
                          (pi(s) - pi(t)).dot(nu.grad(pi(t)) * normal)) /
                         (pi(s) - pi(t)).squaredNorm();
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
          local_integral =
              ComputeDoubleIntegral(integrand, -1, 1, ll, ul, order);
        }

        // Adjacent
        else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
                 (pi(-1) - pi_p(1)).norm() / 100. < tol) {
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

          auto integrand = [&](double phi, double r) {
            // Local arclength coordinates
            double s_pr = r * cos(phi);
            double t_pr = r * sin(phi);
            // Reference interval coordinates
            double s =
                swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
            double t =
                swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;
            // Reference interval coordinates for zeros in arclength coordinates
            double s0 = swap ? -1 : 1;
            double t0 = swap ? 1 : -1;
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(s).norm() * g(pi_p(t)) *
                                  pi_p.Derivative(t).norm() *
                                  (4 / length_pi / length_pi_p);
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            if (r > sqrt_epsilon) {
              // if (r > tol3) {
              singular = (normal.dot(nu(pi(s)) - nu(pi_p(t))) -
                          (pi(s) - pi_p(t)).dot(nu.grad(pi_p(t)) * normal)) /
                         (pi(s) - pi_p(t)).squaredNorm();
            } else {
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
          // Getting the split point for integral
          double alpha = std::atan(length_pi_p / length_pi);
          // Defining upper and lower limits of inner integrals
          auto ll = [&](double phi) { return 0; };
          auto ul1 = [&](double phi) { return length_pi / cos(phi); };
          auto ul2 = [&](double phi) { return length_pi_p / sin(phi); };
          local_integral =
              ComputeDoubleIntegral(integrand, 0, alpha, ll, ul1, order);
          local_integral += ComputeDoubleIntegral(integrand, alpha, M_PI / 2.,
                                                  ll, ul2, order);
        }

        // General
        else {
          auto integrand = [&](double t, double s) {
            // normal
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
          local_integral =
              ComputeDoubleIntegral(integrand, -1, 1, ll, ul, order);
        }

        // Local to global mapping
        unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
        S(II) += local_integral;
      }
    } // loop over pi_p ends
  }
  return -1. / 2. / M_PI * adj_sol.dot(S);
}

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
  bool linear = !true;

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    // evaluating local integrals for all pi_p, with fixed pi and summing
    for (unsigned j = 0; j < numpanels; ++j) {
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {
        double local_integral = 0;
        // coinciding panels
        if (i == j) {
          auto integrand = [&](double t, double s) {
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(t).norm() *
                                  pi.Derivative(s).norm() * g(pi(t));

            Eigen::Vector2d tangent = pi.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            if (fabs(s - t) > sqrt_epsilon) {
              if (!linear) {
                singular = ((pi(s) - pi(t)).dot(normal) *
                            (pi(s) - pi(t)).dot(nu(pi(s)) - nu(pi(t)))) /
                           (pi(s) - pi(t)).squaredNorm() /
                           (pi(s) - pi(t)).squaredNorm();
              } else {
                singular =
                    (pi(s) - pi(t)).dot(normal) / (pi(s) - pi(t)).squaredNorm();
              }
            }

            else {
              if (!linear) {
                singular = 0.5 * pi.DoubleDerivative(t).dot(normal) *
                           pi.Derivative(t).dot(nu.grad(pi(t)).transpose() *
                                                pi.Derivative(t)) /
                           pi.Derivative(t).squaredNorm() /
                           pi.Derivative(t).squaredNorm();
              } else {
                singular = 0.5 * pi.DoubleDerivative(t).dot(normal) /
                           pi.Derivative(t).squaredNorm();
              }
            }

            return singular * non_singular;
          };
          // Function for upper limit of the integral
          auto ul = [&](double x) { return 1.; };
          // Function for lower limit of the integral
          auto ll = [&](double x) { return -1.; };
          local_integral =
              ComputeDoubleIntegral(integrand, -1., 1., ll, ul, order);
        }

        // Adjacent
        else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
                 (pi(-1) - pi_p(1)).norm() / 100. < tol) {
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

          auto integrand = [&](double phi, double r) {
            // Local arclength coordinates
            double s_pr = r * cos(phi);
            double t_pr = r * sin(phi);
            // Reference interval coordinates
            double s =
                swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
            double t =
                swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;
            // Reference interval coordinates for zeros in arclength coordinates
            double s0 = swap ? -1 : 1;
            double t0 = swap ? 1 : -1;
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(s).norm() * g(pi_p(t)) *
                                  pi_p.Derivative(t).norm() *
                                  (4 / length_pi / length_pi_p);
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            if (r > sqrt_epsilon) {
              if (!linear) {
                singular = r *
                           ((pi(s) - pi_p(t)).dot(normal) *
                            (pi(s) - pi_p(t)).dot(nu(pi(s)) - nu(pi_p(t)))) /
                           (pi(s) - pi_p(t)).squaredNorm() /
                           (pi(s) - pi_p(t)).squaredNorm();
              } else {
                singular = r * (pi(s) - pi_p(t)).dot(normal) /
                           (pi(s) - pi_p(t)).squaredNorm();
              }
            } else {
              Eigen::VectorXd vec =
                  2 * cos(phi) * pi.Derivative(s0) / length_pi +
                  2 * sin(phi) * pi_p.Derivative(t0) / length_pi_p;
              vec *= swap ? 1 : -1;

              if (!linear) {
                singular = vec.dot(normal) *
                           (vec.dot(nu.grad(pi(s0)).transpose() * vec)) /
                           (1 + 4 * sin(2 * phi) *
                                    pi.Derivative(s0).dot(pi_p.Derivative(t0)) /
                                    length_pi / length_pi_p) /
                           (1 + 4 * sin(2 * phi) *
                                    pi.Derivative(s0).dot(pi_p.Derivative(t0)) /
                                    length_pi / length_pi_p);
              } else {
                singular = vec.dot(normal) /
                           (1 + 4 * sin(2 * phi) *
                                    pi.Derivative(s0).dot(pi_p.Derivative(t0)) /
                                    length_pi / length_pi_p);
              }
            }
            return singular * non_singular;
          };
          // Getting the split point for integral
          double alpha = std::atan(length_pi_p / length_pi);
          // Defining upper and lower limits of inner integrals
          auto ll = [&](double phi) { return 0; };
          auto ul1 = [&](double phi) { return length_pi / cos(phi); };
          auto ul2 = [&](double phi) { return length_pi_p / sin(phi); };
          local_integral =
              ComputeDoubleIntegral(integrand, 0, alpha, ll, ul1, order);
          // std::cout << "evaluated local integral = " << local_integral <<
          // std::endl;
          local_integral += ComputeDoubleIntegral(integrand, alpha, M_PI / 2.,
                                                  ll, ul2, order);
        }

        // General
        else {
          auto integrand = [&](double t, double s) {
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            if (!linear) {
              return space.evaluateShapeFunction(k, s) *
                     pi.Derivative(s).norm() * pi_p.Derivative(t).norm() *
                     g(pi_p(t)) * (pi(s) - pi_p(t)).dot(normal) *
                     (pi(s) - pi_p(t)).dot(nu(pi(s)) - nu(pi_p(t))) /
                     (pi(s) - pi_p(t)).squaredNorm() /
                     (pi(s) - pi_p(t)).squaredNorm();
            } else {
              return space.evaluateShapeFunction(k, s) *
                     pi.Derivative(s).norm() * pi_p.Derivative(t).norm() *
                     g(pi_p(t)) * (pi(s) - pi_p(t)).dot(normal) /
                     (pi(s) - pi_p(t)).squaredNorm();
            }
          };
          // Function for upper limit of the integral
          auto ul = [&](double x) { return 1.; };
          // Function for lower limit of the integral
          auto ll = [&](double x) { return -1.; };
          local_integral =
              ComputeDoubleIntegral(integrand, -1, 1, ll, ul, order);
        }

        // Local to global mapping
        unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
        T(II) += local_integral;
      }
    } // loop over pi_p ends
  }
  return 1. / M_PI * adj_sol.dot(T);
}

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
    // evaluating local integrals for all pi_p, with fixed pi and summing
    double pi_integral = 0;
    for (unsigned j = 0; j < numpanels; ++j) {

      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {
        double local_integral = 0;
        // coinciding panels
        if (i == j) {
          auto integrand = [&](double t, double s) {
            // std::cout << "(coinciding panels) t,s = " << t << " " << s <<
            // std::endl;
            double non_singular =
                space.evaluateShapeFunction(k, s) * pi.Derivative(t).norm() *
                pi.Derivative(s).norm() *
                (g(pi(t)) * nu.div(pi(t)) + g.grad(pi(t)).dot(nu(pi(t))));

            Eigen::Vector2d tangent = pi.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            if (fabs(s - t) > sqrt_epsilon)
              singular = (pi(s) - pi(t)).dot(normal);
            else
              singular = 0.5 * pi.DoubleDerivative(t).dot(normal) /
                         pi.Derivative(t).squaredNorm();
            return singular * non_singular;
          };
          // Function for upper limit of the integral
          auto ul = [&](double x) { return 1.; };
          // Function for lower limit of the integral
          auto ll = [&](double x) { return -1.; };
          local_integral =
              ComputeDoubleIntegral(integrand, -1., 1., ll, ul, order);
        }

        // Adjacent
        else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
                 (pi(-1) - pi_p(1)).norm() / 100. < tol) {
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

          auto integrand = [&](double phi, double r) {
            // Local arclength coordinates
            double s_pr = r * cos(phi);
            double t_pr = r * sin(phi);
            // Reference interval coordinates
            double s =
                swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
            double t =
                swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;

            // std::cout << "(coinciding panels) t,s = " << t << " " << s <<
            // std::endl;
            double non_singular =
                space.evaluateShapeFunction(k, s) * pi.Derivative(s).norm() *
                pi_p.Derivative(t).norm() * (4 / length_pi / length_pi_p) *
                (g(pi_p(t)) * nu.div(pi_p(t)) +
                 g.grad(pi_p(t)).dot(nu(pi_p(t))));
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal;
            normal << tangent(1), -tangent(0);
            normal /= normal.norm();
            double singular;
            if (r > sqrt_epsilon) {
              singular = r * (pi(s) - pi_p(t)).dot(normal) /
                         (pi(s) - pi_p(t)).squaredNorm();
            } else {
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
          // Getting the split point for integral
          double alpha = std::atan(length_pi_p / length_pi);
          // Defining upper and lower limits of inner integrals
          auto ll = [&](double phi) { return 0; };
          auto ul1 = [&](double phi) { return length_pi / cos(phi); };
          auto ul2 = [&](double phi) { return length_pi_p / sin(phi); };
          local_integral =
              ComputeDoubleIntegral(integrand, 0, alpha, ll, ul1, order);
          local_integral += ComputeDoubleIntegral(integrand, alpha, M_PI / 2.,
                                                  ll, ul2, order);
        }

        // General
        else {
          auto integrand = [&](double t, double s) {
            // normal
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
          local_integral =
              ComputeDoubleIntegral(integrand, -1, 1, ll, ul, order);
        }

        // Local to global mapping
        unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
        U(II) += local_integral;
      }
    } // loop over pi_p ends
  }
  return -1. / 2. / M_PI * adj_sol.dot(U);
}

#endif
