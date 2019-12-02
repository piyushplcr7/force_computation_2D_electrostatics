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
  BoundaryMesh bmesh(coords,elems);
  return bmesh;
}

BoundaryMesh createfrom(const parametricbem2d::ParametrizedMesh &pmesh,
                        unsigned n_i, unsigned n_o) {

  unsigned nV = pmesh.getNumPanels();
  assert(n_i + n_o == nV);
  Eigen::MatrixXd coords(nV, 2);
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> elems(nV, 2);
  for (unsigned i = 0; i < nV; ++i) {
    coords.row(i) = pmesh.getVertex(i);
  }
  for (unsigned i = 0; i < n_i; ++i) {
    elems(i, 0) = i;
    elems(i, 1) = (i + 1) % n_i;
  }
  for (unsigned i = 0; i < n_o; ++i) {
    elems(n_i + i, 0) = n_i + i;
    elems(n_i + i, 1) = n_i + (i + 1) % n_o;
  }
  // std::cout << "mesh coordinates: \n" << coords << std::endl;
  // std::cout << "mesh elements: \n" << elems << std::endl;
  BoundaryMesh bmesh(coords, elems);
  return bmesh;
}

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

int main() {
  std::string fname = "annulardvp.txt";
  std::string fname1 = "neumanncpp.txt";
  std::string fname2 = "neumann2dpbem.txt";
  std::string fname3 = "tnpbemex.txt";
  std::string fname4 = "tncppex.txt";
  std::ofstream out(fname);
  std::ofstream out1(fname1);
  std::ofstream out2(fname2);
  std::ofstream out3(fname3);
  std::ofstream out4(fname4);
  std::cout << std::setw(10) << "panels" << std::setw(10) << "V"
            << std::setw(10) << "K" << std::setw(10) << "M" << std::setw(10)
            << std::endl;

  // Setting up the annular domain
  /*Eigen::MatrixXd cos_list_o(2, 2);
  cos_list_o << 3, 0, 0, 0;
  Eigen::MatrixXd sin_list_o(2, 2);
  sin_list_o << 0, 0, 3, 0;
  Eigen::MatrixXd cos_list_i(2, 2);
  cos_list_i << 1.5, 0, 0, 0;
  Eigen::MatrixXd sin_list_i(2, 2);
  sin_list_i << 0, 0, 1.5, 0;*/
  // annular ellipses
  /*Eigen::MatrixXd cos_list_o(2, 2);
  cos_list_o << 3, 0, 0, 0;
  Eigen::MatrixXd sin_list_o(2, 2);
  sin_list_o << 0, 0, 4, 0;
  Eigen::MatrixXd cos_list_i(2, 2);
  cos_list_i << 2, 0, 0, 0;
  Eigen::MatrixXd sin_list_i(2, 2);
  sin_list_i << 0, 0, 1, 0;*/
  //parametricbem2d::ParametrizedFourierSum outer(cos_list_o, sin_list_o, 0,
  //                                              2 * M_PI);
  //parametricbem2d::ParametrizedFourierSum inner(cos_list_i, sin_list_i, 2*M_PI, 0);
  //double r = 2;
  //parametricbem2d::ParametrizedCircularArc outer(Eigen::Vector2d(2*r,0),4*r,0,2*M_PI);
  //parametricbem2d::ParametrizedCircularArc outer(Eigen::Vector2d(0,0),2*r,0,2*M_PI);
  //parametricbem2d::ParametrizedCircularArc inner(Eigen::Vector2d(0,0),r,2*M_PI,0);

  // Setting up shifted annular domain
  //double r = 2;
  //parametricbem2d::ParametrizedCircularArc outer(Eigen::Vector2d(0,0),r,0,2*M_PI);
  //parametricbem2d::ParametrizedCircularArc inner(Eigen::Vector2d(r/2,0),r/4,2*M_PI,0);

  // Setting up annular square domain
  double a = 1;
  Eigen::Vector2d NE(a/2+0.5,a/2+0.5);
  Eigen::Vector2d NW(-a/2+0.5,a/2+0.5);
  Eigen::Vector2d SE(a/2+0.5,-a/2+0.5);
  Eigen::Vector2d SW(-a/2+0.5,-a/2+0.5);
  // Outer vertices
  Eigen::Vector2d NEo(3*a,3*a);
  Eigen::Vector2d NWo(-3*a,3*a);
  Eigen::Vector2d SEo(3*a,-3*a);
  Eigen::Vector2d SWo(-3*a,-3*a);
  // Inner square
  parametricbem2d::ParametrizedLine ir(NE,SE); // right
  parametricbem2d::ParametrizedLine it(NW,NE); // top
  parametricbem2d::ParametrizedLine il(SW,NW); // left
  parametricbem2d::ParametrizedLine ib(SE,SW); // bottom
  // Outer Square
  parametricbem2d::ParametrizedLine Or(SEo,NEo); // right
  parametricbem2d::ParametrizedLine ot(NEo,NWo); // top
  parametricbem2d::ParametrizedLine ol(NWo,SWo); // left
  parametricbem2d::ParametrizedLine ob(SWo,SEo); // bottom
  // Encoding the dirichlet data for both the boundaries in one function
  double x0 = 0;
  auto potential = [&](double x, double y) {
    // if (x*x+y*y > 2.5) // outer BoundaryMesh
      return x+y;
    // else
    //return x + y; // inner boundary
    //return log(sqrt(x*x+y*y));
    //return log(sqrt((x-x0)*(x-x0)+y*y));
  };
  parametricbem2d::DiscontinuousSpace<0> discont;
  //parametricbem2d::ContinuousSpace<1> cont;

  unsigned maxpanels = 200;
  for (unsigned temp = 1; temp < maxpanels; temp += 3) {
     std::cout << temp << std::endl; // number of panels in both curves
    /*unsigned numpanels_i = temp;
    unsigned numpanels_o = temp;
    parametricbem2d::PanelVector panels_i = inner.split(numpanels_i);
    parametricbem2d::PanelVector panels_o = outer.split(numpanels_o);
    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(), panels_i.begin(),
                  panels_i.end()); // inner panels added first
    panels.insert(panels.end(), panels_o.begin(), panels_o.end());
    parametricbem2d::ParametrizedMesh mesh(panels);*/
    parametricbem2d::PanelVector panels_ir(ir.split(temp));
    parametricbem2d::PanelVector panels_it(it.split(temp));
    parametricbem2d::PanelVector panels_il(il.split(temp));
    parametricbem2d::PanelVector panels_ib(ib.split(temp));

    parametricbem2d::PanelVector panels_or(Or.split(temp));
    parametricbem2d::PanelVector panels_ot(ot.split(temp));
    parametricbem2d::PanelVector panels_ol(ol.split(temp));
    parametricbem2d::PanelVector panels_ob(ob.split(temp));

    parametricbem2d::PanelVector panels;

    panels.insert(panels.end(),panels_ir.begin(),panels_ir.end());
    panels.insert(panels.end(),panels_ib.begin(),panels_ib.end());
    panels.insert(panels.end(),panels_il.begin(),panels_il.end());
    panels.insert(panels.end(),panels_it.begin(),panels_it.end());

    panels.insert(panels.end(),panels_or.begin(),panels_or.end());
    panels.insert(panels.end(),panels_ot.begin(),panels_ot.end());
    panels.insert(panels.end(),panels_ol.begin(),panels_ol.end());
    panels.insert(panels.end(),panels_ob.begin(),panels_ob.end());
    parametricbem2d::ParametrizedMesh mesh(panels);
    //parametricbem2d::ParametrizedMesh mesh_i(panels_i);
    //parametricbem2d::ParametrizedMesh mesh_o(panels_o);
     Eigen::VectorXd sol =
        parametricbem2d::dirichlet_bvp::direct_first_kind::solve(mesh,
                                                                 potential,
                                                                 16);
     std::cout << "Evaluated the 2dpbem sol\n" << sol << std::endl;
    // std::cout << "evaluating V!" << std::endl;
    //parametricbem2d::ParametrizedMesh lmesh = convert_to_linear(mesh);
    //Eigen::MatrixXd K =
    //parametricbem2d::double_layer::GalerkinMatrix(lmesh, cont, discont, 16);
    //std::cout << "K evaluated" << std::endl;
    Eigen::MatrixXd V =
        parametricbem2d::single_layer::GalerkinMatrix(mesh, discont, 16);
        std::cout << "V evaluated \n" << V << std::endl;
    //Eigen::MatrixXd M = parametricbem2d::MassMatrix(lmesh, discont, cont, 16);
    //std::cout << "M evaluated" << std::endl;
    //std::cout << "K 2dpbnem \n" << K << std::endl;
    //std::cout << "M 2dpbnem \n" << M << std::endl;
    //Eigen::MatrixXd K_i =
    //    parametricbem2d::double_layer::GalerkinMatrix(mesh_i, cont,discont, 16);
    //Eigen::MatrixXd K_o =
    //    parametricbem2d::double_layer::GalerkinMatrix(mesh_o, cont,discont, 16);
    //std::cout << "Top left : " << (K.topLeftCorner(temp, temp) - K_i).norm()
    //          << std::endl;
    //std::cout << "bot right : "
    //          << (K.bottomRightCorner(temp, temp) - K_o).norm() << std::endl;
    // std::cout << "V
    // evaluated!" << std::endl;

    // Solving using CPPHilbert
    /*BoundaryMesh bmesh = createfrom(mesh, numpanels_i, numpanels_o);
    //BoundaryMesh bmesh = createfrom(mesh);
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
    }*/
    //std::cout << "Kcpp \n" << Kcpp << std::endl;
    //std::cout << "fabs \n" << (Kcpp-K) << std::endl;
    //std::cout << "Mcpp \n" << Mcpp << std::endl;
    //std::cout << "fabs \n" << (Mcpp-M) << std::endl;
    //Eigen::VectorXd solcpp = Vcpp.lu().solve((0.5 * Mcpp + Kcpp) * g_N);
    //std::cout << std::setw(10) << temp << std::setw(15)
    //          << (V - Vcpp).norm() / Vcpp.norm() << std::setw(15)
    //          << (K - Kcpp).norm() / Kcpp.norm() << std::setw(15)
    //          << (M - Mcpp).norm() / Mcpp.norm() << std::setw(15) << std::endl;

    //Eigen::VectorXd tncpp(maxpanels);
    //Eigen::VectorXd tn2dpbem(maxpanels);
    //Eigen::VectorXd app = Eigen::VectorXd::Constant(maxpanels - 2*temp, 0);
     //tncpp << solcpp, app;
     //tn2dpbem << sol, app;
    //out1 << tncpp.transpose() << std::endl;
    //out2 << tn2dpbem.transpose() << std::endl;

    // Debugging using adnumcse calderon identities, cpp hilbert
    /*Eigen::MatrixXd coords = bmesh.getMeshVertices();
    Eigen::MatrixXi elems = bmesh.getMeshElements();
    Eigen::VectorXd delta_cpp(2*temp); // dirichlet
    Eigen::VectorXd v_cpp(2*temp); // neumann
    for (unsigned I = 0 ; I < 2*temp ; ++I) {
      delta_cpp(I) = potential(coords(I,0),coords(I,1));
      Eigen::Vector2d a = coords.row(elems(I,0));
      Eigen::Vector2d b = coords.row(elems(I,1));
      Eigen::VectorXd tangent = b-a;
      Eigen::Vector2d normal;
      // Outward normal vector
      normal << tangent(1), -tangent(0);
      // Normalizing the normal vector
      normal = normal / normal.norm();
      //normal *= I<temp ? -1:1;
      Eigen::VectorXd gradu(2); gradu << 1,1;
      v_cpp(I) = gradu.dot(normal);
    }*/
    //Eigen::VectorXd tncppex(maxpanels);
    // tncppex << v_cpp, app;
    //out4 << tncppex.transpose() << std::endl;
    //Eigen::VectorXd rho_d_cpp = (0.5*Mcpp+Kcpp)*delta_cpp-Vcpp*v_cpp;
    //std::cout <<std::setw(15)<< rho_d_cpp.lpNorm<Eigen::Infinity>() << std::endl;

    // Debugging using adnumcse calderon identities, 2dpbem
    /*Eigen::VectorXd delta(2*temp); // dirichlet
    Eigen::VectorXd v(2*temp); // neumann
    for (unsigned I = 0 ; I < 2*temp ; ++I) {
      Eigen::Vector2d vertex = mesh.getVertex(I);
      delta(I) = potential(vertex(0),vertex(1));
      Eigen::VectorXd tangent = mesh.getPanels()[I]->Derivative(0.);
      Eigen::Vector2d normal;
      // Outward normal vector
      normal << tangent(1), -tangent(0);
      // Normalizing the normal vector
      normal = normal / normal.norm();
      //normal *= I<temp ? -1:1;
      Eigen::VectorXd gradu(2);
      Eigen::Vector2d point = mesh.getPanels()[I]->operator()(0.);
      double x = point(0); double y = point(1);
      //gradu << 1,1;
      //gradu << x/sqrt(x*x+y*y),y/sqrt(x*x+y*y);
      gradu << (x-x0)/((x-x0)*(x-x0)+y*y),y/((x-x0)*(x-x0)+y*y);
      v(I) = gradu.dot(normal);
    }

    Eigen::VectorXd tnpbemex(maxpanels);
     tnpbemex << v, app;
    out3 << tnpbemex.transpose() << std::endl;*/
    //Eigen::MatrixXd V = parametricbem2d::single_layer::GalerkinMatrix(mesh,discont,16);
    //Eigen::MatrixXd K = parametricbem2d::double_layer::GalerkinMatrix(mesh,cont,discont,16);
    //Eigen::MatrixXd M = parametricbem2d::MassMatrix(mesh, discont, cont, 16);
    //Eigen::VectorXd rho_d = (0.5*M+K)*delta-V*v;
    //std::cout <<std::setw(15)<< rho_d.lpNorm<Eigen::Infinity>() << std::endl;
    //std::cout << std::setw(10) << temp << std::setw(15) << rho_d_cpp.lpNorm<Eigen::Infinity>()
    //          << std::setw(15) << rho_d.lpNorm<Eigen::Infinity>()
    //          << std::endl;
  }
  return 0;
}
