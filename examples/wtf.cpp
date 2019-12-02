#include "BoundaryMesh.hpp"
#include "buildK.hpp"
#include "buildM.hpp"
#include "buildV.hpp"
#include "force_calculation.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_line.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

BoundaryMesh createfrom(const parametricbem2d::ParametrizedMesh& pmesh) {

  unsigned nV = pmesh.getNumPanels();
  Eigen::MatrixXd coords(nV,2);
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> elems(nV,2);
  for (unsigned i = 0 ; i < nV ; ++i) {
    coords.row(i) = pmesh.getVertex(i);
    elems(i,0) = i;
    elems(i,1) = (i+1)%nV;
  }
  //std::cout << "coords \n" << coords << std::endl;
  BoundaryMesh bmesh(coords,elems);
  return bmesh;
}

int main() {
  parametricbem2d::ParametrizedCircularArc curve(Eigen::Vector2d(0, 0), 2,
                                                 0, 2 * M_PI);
   auto potential = [&](double x, double y) {
     //return log(sqrt(x*x+y*y));
     //return x+y;
     return 1;
   };

  for (unsigned numpanels = 2; numpanels < 101; numpanels += 1) {
    std::cout << std::numeric_limits<double>::epsilon() << std::endl;
    parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
    Eigen::VectorXd sol =
       parametricbem2d::dirichlet_bvp::direct_first_kind::solve(mesh,
                                                                potential,
                                                                16);
    //std::cout << "Evaluated the 2dpbem sol\n" << sol << std::endl;
    // Solving using CPPHilbert
    BoundaryMesh bmesh = createfrom(mesh);
    Eigen::MatrixXd Vcpp;
    computeV(Vcpp, bmesh, 0);
    Eigen::SparseMatrix<double> Msp(mesh.getNumPanels(), mesh.getNumPanels());
    computeM01(Msp, bmesh);
    Eigen::MatrixXd Mcpp(Msp);
    Eigen::MatrixXd Kcpp;
    computeK(Kcpp, bmesh, 0);
    Eigen::VectorXd g_N(mesh.getNumPanels());
    for (unsigned i = 0; i < mesh.getNumPanels(); ++i) {
      Eigen::Vector2d pt = mesh.getVertex(i);
      g_N(i) = potential(pt(0), pt(1));
    }
    Eigen::VectorXd solcpp = Vcpp.lu().solve((0.5 * Mcpp + Kcpp) * g_N);
    //std::cout << "solcpp \n" << solcpp << std::endl;
  }
  return 0;
}
