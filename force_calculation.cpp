#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#include "continuous_space.hpp"
#include "dirichlet.hpp"
#include "discontinuous_space.hpp"
#include "gauleg.hpp"
#include "integral_gauss.hpp"
#include "parametrized_mesh.hpp"
#include "single_layer.hpp"
#include <Eigen/Dense>

template <typename G, typename NU>
double CalculateForce(const parametricbem2d::ParametrizedMesh &mesh, const G &g,
                      const NU &nu, unsigned order) {
  double force = 0;
  Eigen::VectorXd state_sol =
      parametricbem2d::dirichlet_bvp::direct_first_kind::solve(mesh, g, order);
  Eigen::VectorXd adj_sol = SolveAdjoint(mesh, g, order);
  force += EvaluateFirst(mesh, g, nu, state_sol, order);
  force += EvaluateSecond();
  force += EvaluateThird();
  force += EvaluateFourth();
  force += EvaluateFifthSixth();
  force += EvaluateSeventh(mesh, g, nu, order);
  return force;
}

template <typename G, typename NU>
double EvaluateFirst(const parametricbem2d::ParametrizedMesh &mesh, const G &g,
                     const NU &nu, const Eigen::VectorXd &state_sol,
                     unsigned order) {
  // The BEM space used for solving the state and adjoint problems
  parametricbem2d::DiscontinuousSpace<0> space;
  // Getting the panels
  PanelVector panels = mesh.getPanels();
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
    for (k = 0; k < q; ++k) {
      // Integrand
      auto integrand = [&](double s) {
        return space.evaluateShapeFunction(k, s) *
               (g.grad(gamma_pi(s)).dot(nu(gamma_pi(s)))) *
               gamma_pi.Derivative(s).squaredNorm();
      };
      // Evaluating the integral
      double integral = ComputeIntegral(integrand, -1, 1, order);
      // Local to global mapping
      unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
      // Adding the integral to the right global place
      V(II) += integral;
    }
  }
  // Evaluating the first term by taking dot product
  return state_sol.dot(V);
}

template <typename G, typename NU>
double EvaluateSeventh(const parametricbem2d::ParametrizedMesh &mesh,
                       const G &g, const NU &nu, const Eigen::VectorXd &adj_sol,
                       unsigned order) {
  // The BEM space used for solving the state and adjoint problems
  parametricbem2d::DiscontinuousSpace<0> space;
  // Getting the panels
  PanelVector panels = mesh.getPanels();
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
    for (k = 0; k < q; ++k) {
      // Integrand
      auto integrand = [&](double s) {
        return space.evaluateShapeFunction(k, s) *
               (g.grad(gamma_pi(s)).dot(nu(gamma_pi(s)))) *
               gamma_pi.Derivative(s).squaredNorm();
      };
      // Evaluating the integral
      double integral = ComputeIntegral(integrand, -1, 1, order);
      // Local to global mapping
      unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
      // Adding the integral to the right global place
      V(II) += integral;
    }
  }
  // Evaluating the first term by taking dot product
  return -0.5 * adj_sol.dot(V);
}

template <typename G, typename NU>
double EvaluateSecond(const parametricbem2d::ParametrizedMesh &mesh,
                      const Eigen::VectorXd &state_sol,
                      const Eigen::VectorXd &adj_sol, const NU &nu,
                      unsigned order) {
  // The BEM space used for solving the state and adjoint problems
  parametricbem2d::DiscontinuousSpace<0> space;
  // Getting the panels
  PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the space
  unsigned q = space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = space.getSpaceDim(numpanels);
  // Initializing the R_{ij} matrix
  Eigen::MatrixXd R(dims, dims);

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    for (unsigned j = 0; j < numpanels; ++j) {

      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];
      double local_integral = 0;

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {
        for (unsigned l = 0; l < q; ++l) {

          // coinciding panels
          if (i == j) {
            auto integrand = [&](s, t) {
              double non_singular = space.evaluateShapeFunction(k, t) *
                                    space.evaluateShapeFunction(l, s) *
                                    pi.Derivative(t).Norm() *
                                    pi.Derivative(s).Norm();
              double singular;
              if (fabs(s - t) > tol)
                singular = (pi(s) - pi(t)).dot(nu(pi(s)) - nu(pi(t))) /
                           (pi(s) - pi(t)).squaredNorm();
              else
                singular =
                    pi.Derivative(t).dot(nu.grad(pi(t)) * pi.Derivative(t)) /
                    (pi.Derivative(t)).squaredNorm();
              return singular * non_singular;
            };
            local_integral = ComputeDoubleIntegral();
          }

          // Adjacent
          else if () {
            bool swap = (pi(1) - pi_p(-1)).norm() / 10. >
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

            auto integrand = [&](double r, double phi) {
              // Local arclength coordinates
              double s_pr = r * cos(phi);
              double t_pr = r * sin(phi);
              // Reference interval coordinates
              double s =
                  swap ? 1 - 2 * s_pr / length_pi : 2 * s_pr / length_pi - 1;
              double t =
                  swap ? 2 * t_pr / length_pi - 1 : 1 - 2 * t_pr / length_pi;
              double non_singular =
                  space.evaluateShapeFunction(k, t) *
                  space.evaluateShapeFunction(l, s) * pi.Derivative(t).Norm() *
                  pi.Derivative(s).Norm() * (-4 / length_pi / length_pi_p);
              double singular;
              if (r > tol) {
                double singular =
                    (pi_p(s) - pi(t)).dot(nu(pi_p(s)) - nu(pi(t))) /
                    (pi_p(s) - pi(t)).squaredNorm();
              } else {
                double singular =
                    (cos(phi) * pi_p.Derivative(s) * 2 / length_pi_p +
                     sin(phi) * pi.Derivative(t) * 2 / length_pi)
                        .dot(nu.grad(pi(t)) *
                             (cos(phi) * pi_p.Derivative(s) * 2 / length_pi_p +
                              sin(phi) * pi.Derivative(t) * 2 / length_pi)) /
                    (1 + sin(2 * phi) * pi.Derivative(t) * pi_p.Derivative(s) *
                             4 / length_pi / length_pi_p);
              }
              return singular * non_singular;
            };
            local_integral = ComputeDoubleIntegral();
          }

          // General
          else {
            auto integrand = [&](s, t) {
              return space.evaluateShapeFunction(k, t) *
                     space.evaluateShapeFunction(l, s) *
                     pi.Derivative(t).Norm() * pi_p.Derivative(s).Norm() *
                     (pi_p(s) - pi(t)).dot(nu(pi_p(s)) - nu(pi(t))) /
                     (pi_p(s) - pi(t)).squaredNorm();
            };
            local_integral = ComputeDoubleIntegral();
          }

          // Local to global mapping
          unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
          unsigned JJ = space.LocGlobMap(l + 1, j + 1, numpanels) - 1;
          R(II, JJ) += local_integral;
        }
      }
    }
  }
  return state_sol.dot(R * adj_sol);
}
