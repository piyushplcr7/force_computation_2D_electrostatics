#ifndef FORCECALCULATIONHPP
#define FORCECALCULATIONHPP

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
#include "parametrized_mesh.hpp"
#include "single_layer.hpp"
#include <Eigen/Dense>

/**
 * This function is used to compute double integrals of the form:
 * /f$ \int_{x=a}^{b}\int{y=ll(x)}^{ul(x)} f(x,y) dy dx /f$
 */
template <typename F, typename LL, typename UL>
double ComputeDoubleIntegral(const F &f, double a, double b, const LL &ll,
                             const UL &ul, unsigned order) {
  //
  double integral = 0;
  // We need the function fx(x) which is obtained from the integral over y.
  auto fx = [&](double x) {
    auto temp = [&](double y) { return f(x, y); };
    double a = ll(x);
    double b = ul(x);
    return ComputeIntegral(temp, a, b, order);
  };
  return ComputeIntegral(fx, a, b, order);
}

/*Eigen::VectorXd SolveAdjoint(const parametricbem2d::ParametrizedMesh &mesh,
                             std::function<double(double, double)> g,
                             unsigned order) {
  // Trial and test space
  parametricbem2d::DiscontinuousSpace<0> trial_space;
  // Getting the LHS
  Eigen::MatrixXd V =
      parametricbem2d::single_layer::GalerkinMatrix(mesh, trial_space, order);
  // Calculating the rhs vector
  // Number of reference shape functions for the space
  unsigned q = space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = space.getSpaceDim(numpanels);
  // Initializing the rhs vector
  Eigen::VectorXd rhs = Eigen::VectorXd::Constant(dims, 0);
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &gamma_pi = *panels[i];
    for (unsigned k = 0; k < q; ++k) {
      auto integrand = [&](double s) {
        return trial_space.evaluateShapeFunction(k, s) *
               g(gamma_pi(s)(0), gamma_pi(s)(1)) * gamma_pi.Derivative(s).norm();
      };
      // Evaluating the local integral
      double local_integral = ComputeIntegral(integral,-1,1,order);
      // Local to global map
      unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
      rhs(II) += integral;
    }
  }
  Eigen::FullPivLU<Eigen::MatrixXd> dec(V);
  Eigen::VectorXd sol = dec.solve(rhs);
  return sol;
}*/

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
                                    pi.Derivative(t).norm() *
                                    pi.Derivative(s).norm();
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
              double s = swap ? 1 - 2 * s_pr / length_pi_p
                              : 2 * s_pr / length_pi_p - 1;
              double t =
                  swap ? 2 * t_pr / length_pi - 1 : 1 - 2 * t_pr / length_pi;
              double non_singular =
                  space.evaluateShapeFunction(k, t) *
                  space.evaluateShapeFunction(l, s) * pi.Derivative(t).norm() *
                  pi_p.Derivative(s).norm() * (4 / length_pi / length_pi_p);
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
                     pi.Derivative(t).norm() * pi_p.Derivative(s).norm() *
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
  return -1 / 2. / M_PI * state_sol.dot(R * adj_sol);
}

template <typename G, typename NU>
double EvaluateThird(const parametricbem2d::ParametrizedMesh &mesh,
                     const Eigen::VectorXd &adj_sol, const G &g, const NU &nu,
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
  // Initializing the S_{i} vector
  Eigen::VectorXd S = Eigen::VectorXd::Constant(dims, 0);

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    // evaluating local integrals for all pi_p, with fixed pi and summing
    double pi_integral = 0;
    for (unsigned j = 0; j < numpanels; ++j) {
      double local_integral = 0;
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {

        // coinciding panels
        if (i == j) {
          auto integrand = [&](s, t) {
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(t).norm() *
                                  pi.Derivative(s).norm() * g(pi(t));

            Eigen::Vector2d tangent = pi.Derivative(t);
            Eigen::Vector2d normal(tangent(1), -tangent(0));
            normal /= normal.norm();
            double singular;
            if (fabs(s - t) > tol)
              singular =
                  (normal.dot(nu(pi(s)) - nu(pi(t))) -
                   (pi(s) - pi(t)).dot(nu.grad(pi(t)).transpose() * normal)) /
                  (pi(s) - pi(t)).squaredNorm();
            else
              singular =
                  0.5 *
                  (normal.dot(nu.dgrad1(pi(t)) *
                                  pi.Derivative(t).dot(pi.Derivative(t)) +
                              nu.dgrad2(pi(t)) *
                                  pi.Derivative(t).dot(pi.Derivative(t)))) /
                  pi.Derivative(t).squaredNorm();
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
                swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
            double t =
                swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(s).norm() * g(pi_p(t)) *
                                  pi_p.Derivative(t).norm() *
                                  (4 / length_pi / length_pi_p);
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal(tangent(1), -tangent(0));
            normal /= normal.norm();
            double singular;
            if (r > tol) {
              double singular =
                  (normal.dot(nu(pi(s)) - nu(pi_p(t))) -
                   (pi(s) - pi_p(t))
                       .dot(nu.grad(pi_p(t)).transpose() * normal)) /
                  (pi(s) - pi_p(t)).squaredNorm();
            } else {
              Eigen::Vector2d vec1(
                  pi.Derivative(s).dot(nu.dgrad1(pi(s)) * pi.Derivative(s)),
                  pi.Derivative(s).dot(nu.dgrad2(pi(s)) * pi.Derivative(s)));
              Eigen::Vector2d vec2(pi_p.Derivative(t).dot(nu.dgrad1(pi_p(t)) *
                                                          pi_p.Derivative(t)),
                                   pi_p.Derivative(t).dot(nu.dgrad2(pi_p(t)) *
                                                          pi_p.Derivative(t)));
              Eigen::Vector2d vec3(
                  pi_p.Derivative(t).dot(nu.dgrad1(pi_p(t)) * pi.Derivative(s)),
                  pi_p.Derivative(t).dot(nu.dgrad2(pi_p(t)) *
                                         pi.Derivative(s)));
              double singular =
                  (normal.dot(2 * cos(phi) * cos(phi) * vec1 / length_pi /
                                  length_pi +
                              2 * sin(phi) * sin(phi) * vec2 / length_pi_p /
                                  length_pi_p +
                              4 * sin(phi) * cos(phi) * vec3 / length_pi /
                                  length_pi_p)) /
                  (1 + 4 * sin(2 * phi) *
                           pi.Derivative(s).dot(pi_p.Derivative(t)) /
                           length_pi / length_pi_p);
            }
            return singular * non_singular;
          };
          local_integral = ComputeDoubleIntegral();
        }

        // General
        else {
          auto integrand = [&](s, t) {
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal(tangent(1), -tangent(0));
            normal /= normal.norm();
            return space.evaluateShapeFunction(k, s) * pi.Derivative(s).norm() *
                   pi_p.Derivative(t).norm() * g(pi_p(t)) *
                   (normal.dot(nu(pi(s)) - nu(pi_p(t))) -
                    (pi(s) - pi_p(t))
                        .dot(nu.grad(pi_p(t)).transpose() * normal)) /
                   (pi(s) - pi_p(t)).squaredNorm();
          };
          local_integral = ComputeDoubleIntegral();
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
  PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the space
  unsigned q = space.getQ();
  // Getting the number of panels
  unsigned numpanels = mesh.getNumPanels();
  // Getting space dimensions
  unsigned dims = space.getSpaceDim(numpanels);
  // Initializing the T_{i} vector
  Eigen::VectorXd T = Eigen::VectorXd::Constant(dims, 0);

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    // evaluating local integrals for all pi_p, with fixed pi and summing
    double pi_integral = 0;
    for (unsigned j = 0; j < numpanels; ++j) {
      double local_integral = 0;
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {

        // coinciding panels
        if (i == j) {
          auto integrand = [&](s, t) {
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(t).norm() *
                                  pi.Derivative(s).norm() * g(pi(t));

            Eigen::Vector2d tangent = pi.Derivative(t);
            Eigen::Vector2d normal(tangent(1), -tangent(0));
            normal /= normal.norm();
            double singular;
            if (fabs(s - t) > tol)
              singular = ((pi(s) - pi(t)).dot(normal) *
                          (pi(s) - pi(t)).dot(nu(pi(s)) - nu(pi(t)))) /
                         (pi(s) - pi(t)).squaredNorm() /
                         (pi(s) - pi(t)).squaredNorm();
            else
              singular =
                  0.5 * pi.DoubleDerivative(t).dot(normal) *
                  pi.Derivative(t).dot(nu.grad(pi(t)) * pi.Derivative(t)) /
                  pi.Derivative(t).squaredNorm() /
                  pi.Derivative(t).squaredNorm();
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
                swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
            double t =
                swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;
            double non_singular = space.evaluateShapeFunction(k, s) *
                                  pi.Derivative(s).norm() * g(pi_p(t)) *
                                  pi_p.Derivative(t).norm() *
                                  (4 / length_pi / length_pi_p);
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal(tangent(1), -tangent(0));
            normal /= normal.norm();
            double singular;
            if (r > tol) {
              double singular =
                  ((pi(s) - pi_p(t)).dot(normal) *
                   (pi(s) - pi_p(t)).dot(nu(pi(s)) - nu(pi_p(t)))) /
                  (pi(s) - pi_p(t)).squaredNorm() /
                  (pi(s) - pi_p(t)).squaredNorm();
            } else {
              Eigen::Vector2d vec(2 * cos(phi) * pi.Derivative(s) / length_pi +
                                  2 * sin(phi) * pi_p.Derivative(t) /
                                      length_pi_p);
              vec1 *= swap ? 1 : -1;

              double singular =
                  vec.dot(normal) * (vec.dot(nu.grad(pi(s)) * vec)) /
                  (1 + 4 * sin(2 * phi) *
                           pi.Derivative(s).dot(pi_p.Derivative(t)) /
                           length_pi / length_pi_p) /
                  (1 + 4 * sin(2 * phi) *
                           pi.Derivative(s).dot(pi_p.Derivative(t)) /
                           length_pi / length_pi_p);
            }
            return singular * non_singular;
          };
          local_integral = ComputeDoubleIntegral();
        }

        // General
        else {
          auto integrand = [&](s, t) {
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal(tangent(1), -tangent(0));
            normal /= normal.norm();
            return space.evaluateShapeFunction(k, s) * pi.Derivative(s).norm() *
                   pi_p.Derivative(t).norm() * g(pi_p(t)) *
                   (pi(s) - pi_p(t)).dot(normal) *
                   (pi(s) - pi_p(t)).dot(nu(pi(s)) - nu(pi_p(t))) /
                   (pi(s) - pi_p(t)).squaredNorm() /
                   (pi(s) - pi_p(t)).squaredNorm();
          };
          local_integral = ComputeDoubleIntegral();
        }

        // Local to global mapping
        unsigned II = space.LocGlobMap(k + 1, i + 1, numpanels) - 1;
        T(II) += local_integral;
      }
    } // loop over pi_p ends
  }
  return 1. / M_PI * adj_sol.dot(S);
}

template <typename G, typename NU>
double EvaluateFiveSix(const parametricbem2d::ParametrizedMesh &mesh,
                       const Eigen::VectorXd &adj_sol, const G &g, const NU &nu,
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
  // Initializing the U_{i} vector
  Eigen::VectorXd U = Eigen::VectorXd::Constant(dims, 0);

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels; ++i) {
    // evaluating local integrals for all pi_p, with fixed pi and summing
    double pi_integral = 0;
    for (unsigned j = 0; j < numpanels; ++j) {
      double local_integral = 0;
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {

        // coinciding panels
        if (i == j) {
          auto integrand = [&](s, t) {
            double non_singular =
                space.evaluateShapeFunction(k, s) * pi.Derivative(t).norm() *
                pi.Derivative(s).norm() *
                (g(pi(t)) * nu.div(pi(t)) + g.grad(pi(t)).dot(nu(pi(t))));

            Eigen::Vector2d tangent = pi.Derivative(t);
            Eigen::Vector2d normal(tangent(1), -tangent(0));
            normal /= normal.norm();
            double singular;
            if (fabs(s - t) > tol)
              singular = ((pi(s) - pi(t)).dot(normal);
            else
              singular = 0.5 * pi.DoubleDerivative(t).dot(normal) /
                         pi.Derivative(t).squaredNorm();
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
                swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
            double t =
                swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;
            double non_singular =
                space.evaluateShapeFunction(k, s) * pi.Derivative(s).norm() *
                pi_p.Derivative(t).norm() * (4 / length_pi / length_pi_p) *
                (g(pi_p(t)) * nu.div(pi_p(t)) +
                 g.grad(pi_p(t)).dot(nu(pi_p(t))));
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal(tangent(1), -tangent(0));
            normal /= normal.norm();
            double singular;
            if (r > tol) {
              double singular =
                  ((pi(s) - pi_p(t)).dot(normal)/
                  (pi(s) - pi_p(t)).squaredNorm();
            } else {
              Eigen::Vector2d vec(2 * cos(phi) * pi.Derivative(s) / length_pi +
                                  2 * sin(phi) * pi_p.Derivative(t) /
                                      length_pi_p);
              vec1 *= swap ? 1 : -1;

              double singular =
                  vec.dot(normal) /
                  (1 + 4 * sin(2 * phi) *
                           pi.Derivative(s).dot(pi_p.Derivative(t)) /
                           length_pi / length_pi_p)
            }
            return singular * non_singular;
          };
          local_integral = ComputeDoubleIntegral();
        }

        // General
        else {
          auto integrand = [&](s, t) {
            // normal
            Eigen::Vector2d tangent = pi_p.Derivative(t);
            Eigen::Vector2d normal(tangent(1), -tangent(0));
            normal /= normal.norm();
            return space.evaluateShapeFunction(k, s) * pi.Derivative(s).norm() *
                   pi_p.Derivative(t).norm() *
                   (g(pi_p(t)) * nu.div(pi_p(t)) +
                    g.grad(pi_p(t)).dot(nu(pi_p(t)))) *
                   (pi(s) - pi_p(t)).dot(normal) /
                   (pi(s) - pi_p(t)).squaredNorm();
          };
          local_integral = ComputeDoubleIntegral();
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
