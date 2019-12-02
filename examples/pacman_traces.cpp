#include "BoundaryMesh.hpp"
#include "buildK.hpp"
#include "buildM.hpp"
#include "buildV.hpp"
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

BoundaryMesh createfrom(const parametricbem2d::ParametrizedMesh& pmesh) {

  unsigned nV = pmesh.getNumPanels();
  Eigen::MatrixXd coords(nV,2);
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> elems(nV,2);
  for (unsigned i = 0 ; i < nV ; ++i) {
    coords.row(i) = pmesh.getVertex(i);
    elems(i,0) = i;
    elems(i,1) = (i+1)%nV;
  }
  BoundaryMesh bmesh(coords,elems);
  std::cout << "coords \n" << coords << std::endl;
  std::cout << "elems \n" << elems << std::endl;
  return bmesh;
}

int main() {
  std::cout << xy_to_phi(0.5,0) << std::endl;
  // pacman
  Eigen::Vector2d B(0, -1);
  Eigen::Vector2d R(1, 0);
  Eigen::Vector2d C(0, 0);

  parametricbem2d::ParametrizedLine l1(C, R); // right
  parametricbem2d::ParametrizedLine l2(B, C); // top
  parametricbem2d::ParametrizedCircularArc curve(Eigen::Vector2d(0,0),1,0,3*M_PI/2);

  unsigned order = 16;

  auto G = [&](double x, double y) {
    double r = std::sqrt(x*x+y*y);
    double phi = xy_to_phi(x,y);
    return std::pow(r,2./3.)*sin(2./3.*phi);
  };

  auto GS = [&](double x, double y) {
    double r = std::sqrt(x*x+y*y);
    double phi = xy_to_phi(x,y);
    return r*sin(2./3.*phi);
  };

  auto GNS = [&](double x, double y) {
    double r = std::sqrt(x*x+y*y);
    if (r<1) {
      return 0.;
    }
    else {
      double phi = xy_to_phi(x,y);
      return sin(2./3.*phi);
    }
  };

  for (unsigned numpanels = 2; numpanels < 1001; numpanels += 1) {
    //auto start = std::chrono::system_clock::now();
    unsigned temp = numpanels;
    parametricbem2d::PanelVector panels_l1(l1.split(temp));
    parametricbem2d::PanelVector panels_curve(curve.split(temp));
    parametricbem2d::PanelVector panels_l2(l2.split(temp));

    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(), panels_l1.begin(), panels_l1.end());
    panels.insert(panels.end(), panels_curve.begin(), panels_curve.end());
    panels.insert(panels.end(), panels_l2.begin(), panels_l2.end());
    parametricbem2d::ParametrizedMesh mesh(panels);

    BoundaryMesh bmesh = createfrom(mesh);
    Eigen::MatrixXd Vcpp;
    computeV(Vcpp, bmesh, 0);
    //std::cout << "VCpp \n" << Vcpp << std::endl;
    Eigen::SparseMatrix<double> Msp(mesh.getNumPanels(), mesh.getNumPanels());
    computeM01(Msp, bmesh);
    Eigen::MatrixXd Mcpp(Msp);
    //std::cout << " MCpp \n" << Mcpp << std::endl;
    Eigen::MatrixXd Kcpp;
    computeK(Kcpp, bmesh, 0);
    //std::cout << "KCpp \n" << Kcpp << std::endl;
    Eigen::VectorXd g_N(mesh.getNumPanels());
    for (unsigned i = 0; i < mesh.getNumPanels(); ++i) {
      Eigen::Vector2d pt = mesh.getVertex(i);
      g_N(i) = G(pt(0), pt(1));
    }
    std::cout << "gn\n" << g_N << std::endl;
    Eigen::VectorXd solcpp = Vcpp.lu().solve((0.5 * Mcpp + Kcpp) * g_N);

    Eigen::VectorXd sol =
       parametricbem2d::dirichlet_bvp::direct_first_kind::solve(mesh,
                                                                GNS,
                                                                16);
    std::cout << "Pbem trace: for numpanels = " << mesh.getNumPanels() << std::endl;
    std::cout << solcpp << std::endl;
  }

  return 0;
}
