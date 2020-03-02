#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "shape_calculus.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
  std::string fname = "quad_approx_circle.txt";
  std::string fname1 = "lowrank_approx_circle.txt";
  std::string fname2 = "eigvecs.txt";
  std::ofstream out(fname);
  std::ofstream out1(fname1);
  std::ofstream out2(fname2);
  double R = 1.5;
  // Definition of the domain
  parametricbem2d::ParametrizedCircularArc domain(Eigen::Vector2d(0, 0), R, 0,
                                                  2 * M_PI);

  unsigned order = 16;

  // Defining the BEM space
  parametricbem2d::ContinuousSpace<1> space;

  // Definition of the shape functional
  auto F = [&](Eigen::Vector2d X) {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d(x * x * x, y * y * y);
    //return Eigen::Vector2d(x * x * x, y * y);
  };

  class func {
  public:
    double operator()(const Eigen::Vector2d &X) const {
      double x = X(0);
      double y = X(1);
      return 3 * x * x + 3 * y * y;
      //return 3 * x * x + 2 * y;
    }

    Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
      double x = X(0);
      double y = X(1);
      return Eigen::Vector2d(6 * x, 6 * y);
      //return Eigen::Vector2d(3 * x, 2);
    }
  };

  func f;

  auto shape_functional = [&](const parametricbem2d::ParametrizedMesh &mesh) {
    unsigned numpanels = mesh.getNumPanels();
    parametricbem2d::PanelVector panels = mesh.getPanels();
    double val = 0;
    for (unsigned i = 0; i < numpanels; ++i) {
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      auto integrand = [&](double s) {
        Eigen::Vector2d tangent = pi.Derivative(s);
        Eigen::Vector2d normal;
        normal << tangent(1), -tangent(0);
        normal /= normal.norm();
        return F(pi(s)).dot(normal) * pi.Derivative(s).norm();
      };
      val += parametricbem2d::ComputeIntegral(integrand, -1, 1, order);
    }
    return val;
  };

  auto vel_field = [&](double x, double y) {
     return cos(x) * cos(y);
    //return x * x * x * y * y;
  };

  std::function<double(double, double)> d_cont = vel_field;

  /*unsigned rlevs = 20;
  Eigen::ArrayXd rs = Eigen::ArrayXd::LinSpaced(rlevs, R - 0.3, R + 0.3);
  Eigen::ArrayXd deltars = rs + (-R);
  Eigen::ArrayXd err = Eigen::ArrayXd::Constant(rlevs, 0);

  // Base level shape functional check
  double base_sf = shape_functional(mesh);
  double base_sf_ex = 1.5 * M_PI * std::pow(R, 4);
  std::cout << "shape functional : " << base_sf
            << " , exact formula: " << base_sf_ex << std::endl;
  for (unsigned i = 0; i < rlevs; ++i) {
    // Definition of the displacement vector field
    auto d = [&](const Eigen::Vector2d &X) {
      double x = X(0);
      double y = X(1);
      return deltars(i);
    };

    std::function<double(Eigen::Vector2d)> D = d;
    // Getting the shape gradient and hessian value for the displacement
    double shape_gradient = shapeGradient(mesh, f, D, order);
    double shape_hessian = shapeHessian(mesh, f, D, order);
    double quad_approx = base_sf_ex + shape_gradient + shape_hessian;
    double sf_ex = 1.5 * M_PI * std::pow(rs(i), 4);
    err(i) = fabs(sf_ex - quad_approx) / fabs(sf_ex);
  }
  Eigen::MatrixXd output(rlevs, 2);
  output << deltars, err;
  out.precision(std::numeric_limits<double>::digits10);
  out << std::setw(5) << "#deltars" << std::setw(15) << "error" << std::endl;
  out << output << std::endl;*/

  // numpanels 11 gives a nan eigenvector!?
  unsigned start = 4;
  unsigned maxpanels = 50; // start+1;
  double t = 0.5;
  // double SF0 = 1.5 * M_PI * std::pow(R, 4);
  // std::cout << "SF0 " << SF0 << std::endl;
  // double SFt = 1.5 * M_PI * std::pow(R + t, 4);
  // std::cout << "SFt " << SFt << std::endl;
  Eigen::MatrixXd qa = Eigen::VectorXd::Constant(maxpanels, 0);
  Eigen::MatrixXd lra = Eigen::MatrixXd::Constant(maxpanels, maxpanels, 0);

  for (unsigned numpanels = start; numpanels < maxpanels; numpanels += 1) {
    std::cout << "numpanels " << numpanels << std::endl;
    unsigned dim = space.getSpaceDim(numpanels);
    parametricbem2d::ParametrizedMesh mesh(domain.split(numpanels));

    // double ex_calc = shape_functional(mesh);
    // std::cout << "SF0 calc: " << ex_calc << std::endl;

    // Velocity field
    // Eigen::VectorXd d = t * Eigen::VectorXd::Constant(dim, 1);
     Eigen::VectorXd d = space.Interpolate(vel_field, mesh);
    //Eigen::VectorXd d = Eigen::VectorXd::Random(dim);

    // Getting the relevant matrices
    auto bundle = getRelevantQuantities(mesh, f, order, space);
    Eigen::MatrixXd Mg = bundle.first.first;
    Eigen::MatrixXd A_pr = bundle.first.second;
    Eigen::VectorXd V = bundle.second;

    // Calculating d*, which represents the linear part in the quadratic
    // functional
    Eigen::VectorXd dstar = Mg.lu().solve(V / 2.);
    // Calculating the quadratic approximation
    double quad_approx =
        (d + dstar).dot(Mg * (d + dstar)) - dstar.dot(Mg * dstar); // + SF0;
    qa(numpanels) = quad_approx;

    // Solving the general eigenvalue problem
    // Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
    ges.compute(Mg, A_pr);
    Eigen::VectorXd eigvals = ges.eigenvalues().real();     // eigenvalues
    Eigen::MatrixXd eigvectors = ges.eigenvectors().real(); // eigenvectors

    // To check for NaNs in the eigensolution
    //std::cout << eigvectors.sum() << std::endl;

    // Storing the eigenvalues as std vector for sorting
    std::vector<double> temp(eigvals.data(), eigvals.data() + dim);
    // Sorting while storing the indices
    auto idx = sort_indexes(temp);

    for (unsigned K = 1; K < dim; ++K) {
      std::cout << "K: " << K << std::endl;

      Eigen::MatrixXd eigvectors_K(dim, K);
      eigvectors_K.col(0) = dstar;
      // Storing the top K eigenvectors
      for (unsigned i = 1; i < K; ++i) {
        eigvectors_K.col(i) = eigvectors.col(idx[i-1]);
      }
      // Doing Gram Schmidt orthogonalization on the chosen eigenvectors
      Eigen::MatrixXd ortho = GramSchmidtOrtho(eigvectors_K, A_pr);

      // Low rank projection for d+d*
      Eigen::VectorXd dpds_proj =
          ortho * ortho.transpose() * A_pr * (d + dstar);
      double low_rank_approx =
          dpds_proj.dot(Mg * dpds_proj) - dstar.dot(Mg * dstar); // + SF0;

      lra(numpanels, K) = low_rank_approx;

      if (numpanels == maxpanels-1 && K==dim-1) {
        out2.precision(std::numeric_limits<double>::digits10);
        out2 << eigvectors_K << std::endl;
      }
    } // Loop over K

    if (!true) {
      unsigned K = 3;
      double abs_err = compare(mesh, f, d, K, order, space);
      std::cout << "abs err: " << abs_err << std::endl;
    }

  } // Loop over numpanels
  out.precision(std::numeric_limits<double>::digits10);
  out << qa << std::endl;
  out1.precision(std::numeric_limits<double>::digits10);
  out1 << lra << std::endl;

  return 0;
}
