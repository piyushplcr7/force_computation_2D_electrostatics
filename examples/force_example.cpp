#include "BoundaryMesh.hpp"
#include "force_calculation.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>

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

// format of the gradient:
// dv1/dx1  dv2/dx1
// dv1/dx2  dv2/dx2

// format of dgrad:
// d2 v/dx1 dx1              d2 v/ dx1 dx2
// d2 v/ dx1 dx2             d2 v/ dx2 dx2

double ri = 1.5;
double ro = 3;

double R = 1.5; // 2;

class NU {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    return Eigen::Vector2d(-y / r, x / r);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    double r3 = r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << x * y / r3, -x * x / r3, y * y / r3, -x * y / r3;
    return M.transpose();
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    return 0;
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    double r5 = r * r * r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << y * (y * y - 2 * x * x) / r5, x * (x * x - 2 * y * y) / r5,
        x * (x * x - 2 * y * y) / r5, 3 * x * x * y / r5;
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    double r5 = r * r * r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << -3 * x * y * y / r5, y * (2 * x * x - y * y) / r5,
        y * (2 * x * x - y * y) / r5, x * (2 * y * y - x * x) / r5;
    return M;
  }
};

class NU_ELLIPSE {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d(-y / 2., 2 * x);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 0, -0.5, 2, 0;
    return M.transpose();
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return 0;
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
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_CIRCLE {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d(-y, x);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 0, -1, 1, 0;
    return M.transpose();
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return 0;
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
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_LINEAR {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const { return X; }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 1, 0, 0, 1;
    return M.transpose();
  }

  double div(const Eigen::Vector2d &X) const { return 2; }

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
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_CONST {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    Eigen::Vector2d out;
    out << 1, 1;
    return out * 0.5;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const { return 0; }

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
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_RADIAL_APP {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    if (x * x + y * y < 1.1 * 1.1 * R * R)
      out << x + x * x + y * y - R * R, y + x * x + y * y - R * R;
    else
      out << x + x * x + y * y - 4 * R * 4 * R,
          y + x * x + y * y - 4 * R * 4 * R;
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 1 + 2 * x, 2 * x, 2 * y, 1 + 2 * y;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return 2 * (1 + x + y);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 2, 0, 0, 2;
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << 2, 0, 0, 2;
    return M;
  }
};

class NU_RADIAL {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    out << x, y;
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 1, 0, 0, 1;
    return M;
  }

  double div(const Eigen::Vector2d &X) const { return 2; }

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
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
};

class NU_RADIAL_NORM {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    Eigen::Vector2d out;
    out << x / r, y / r;
    // if (x*x+y*y < 1.1*1.1*R*R)
    return out;
    // else
    // return Eigen::Vector2d(0,0);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r3 = r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << y * y / r3, -x * y / r3, -x * y / r3, x * x / r3;
    // if (x*x+y*y < 1.1*1.1*R*R)
    return M;
    // else
    // return Eigen::MatrixXd::Constant(2,2,0);
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    // if (x*x+y*y < 1.1*1.1*R*R)
    return 1 / r;
    // else
    // return 0;
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r5 = r * r * r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << -3 * x * y * y / r5, y * (2 * x * x - y * y) / r5,
        y * (2 * x * x - y * y) / r5, x * (2 * y * y - x * x) / r5;
    // if (x*x+y*y < 1.1*1.1*R*R)
    return M;
    // else
    // return Eigen::MatrixXd::Constant(2,2,0);
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r5 = r * r * r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << y * (2 * x * x - y * y) / r5, x * (2 * y * y - x * x) / r5,
        x * (2 * y * y - x * x) / r5, -3 * x * x * y / r5;
    // if (x*x+y*y < 1.1*1.1*R*R)
    return M;
    // else
    // return Eigen::MatrixXd::Constant(2,2,0);
  }
};

class G {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    // return 1;
    return x + y;
    // return log(sqrt(x * x + y * y));
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return Eigen::Vector2d(1, 1);
    // return Eigen::Vector2d(0,0);
    // return Eigen::Vector2d(x / (x * x + y * y), y / (x * x + y * y));
  }
};

class G_SIMON {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double phi = atan(y / x);
    if (x * x + y * y < 1.1 * 1.1 * R * R)
      // return 0.5*(3+2*(x*x)/(ri*ri)-1+2*y/ri);
      return log(R);
    // return log(R)*R/sqrt(x*x+y*y);
    // return 0;
    // return log(sqrt(x * x + y * y));
    // return log(sqrt((x-0.8) * (x-0.8) + (y-0.5) * (y-0.5)));
    // return log(sqrt(x * x + y * y))+x*x+y*y-R*R;
    // return log(R);//+x*x+y*y-R*R;
    // return x+y+x*x+y*y-R*R;
    else
      return log(sqrt(x * x + y * y));
    // return log(sqrt((x-0.8) * (x-0.8) + (y-0.5) * (y-0.5)));
    // return log(sqrt(x * x + y * y))+(x-2*R)*(x-2*R)+y*y-4*R*4*R;
    // return 1;
    // log(4*R);//+x*x+y*y-4*R*4*R;
    // return x+y+x*x+y*y-4*R*4*R;

    // if ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5) < 1.1 * 1.1 * R * R)
    // return log(sqrt((x-0.8) * (x-0.8) + (y-0.5) * (y-0.5)));
    // return log(R);
    // else
    // return log(sqrt((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5)));
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    if (x * x + y * y < 1.1 * 1.1 * R * R)
      return Eigen::Vector2d(0, 0);
    // return log(R)*R*Eigen::Vector2d(-x/r/r/r,-y/r/r/r);
    //  return Eigen::Vector2d(2*x/ri/ri,1./ri);
    // return Eigen::Vector2d((x-0.8) / ((x-0.8) * (x-0.8) + (y-0.5) * (y-0.5)),
    //                       (y-0.5) / ((x-0.8) * (x-0.8) + (y-0.5) * (y-0.5)));
    // return Eigen::Vector2d(x / (x * x + y * y), y / (x * x + y * y));
    // return Eigen::Vector2d(x / (x * x + y * y)+2*x, y / (x * x + y * y)+2*y);
    // return Eigen::Vector2d(2*x,2*y);
    // return Eigen::Vector2d(1+2*x,1+2*y);
    else
      // return Eigen::Vector2d(0,0);
      // return Eigen::Vector2d(2*x,2*y);
      // return Eigen::Vector2d(1+2*x,1+2*y);
      // return Eigen::Vector2d(x / (x * x + y * y)+2*(x-2*R), y / (x * x + y *
      // y)+2*y);
      return Eigen::Vector2d(x / (x * x + y * y), y / (x * x + y * y));
    // return Eigen::Vector2d((x-0.8) / ((x-0.8) * (x-0.8) + (y-0.5) * (y-0.5)),
    //                       (y-0.5) / ((x-0.8) * (x-0.8) + (y-0.5) * (y-0.5)));

    // if ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5) < 1.1 * 1.1 * R * R)
    // return Eigen::Vector2d((x-0.8) / ((x-0.8) * (x-0.8) + (y-0.5) *
    // (y-0.5)),
    //                     (y-0.5) / ((x-0.8) * (x-0.8) + (y-0.5) * (y-0.5)));
    // return Eigen::Vector2d(0, 0);
    // else
    //  return Eigen::Vector2d(
    //  (x - 0.8) / ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5)),
    //  (y - 0.5) / ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5)));
  }
};

class NU_Y_SIMON {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    if ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5) <
        1.1 * 1.1 * R * R) { //(x * x + y * y < 1.1 * 1.1 * R * R) {
                             // out << 1, 0;
      out << x - 0.8, 0;
      // out << 1,0;
    } else {
      out << 0, 0;
      // out << x-0.8, 0;
      // out << 1, 0;
      // out << x, 0;
    }
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    if ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5) <
        1.1 * 1.1 * R * R) { //(x * x + y * y < 1.1 * 1.1 * R * R) {
      M << 1, 0, 0, 0;
      // M << 0, 0, 0, 0;
    } else {
      M << 0, 0, 0, 0;
      // M << 1, 0, 0, 0;
    }
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    if ((x - 0.8) * (x - 0.8) + (y - 0.5) * (y - 0.5) <
        1.1 * 1.1 * R * R) { //(x * x + y * y < 1.1 * 1.1 * R * R) {
      return 1;
      // return 0;
    } else {
      return 0;
      // return 1;
    }
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    if (x * x + y * y < 1.1 * 1.1 * R * R) {
      M << 0, 0, 0, 0;
    } else {
      M << 0, 0, 0, 0;
    }
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    if (x * x + y * y < 1.1 * 1.1 * R * R) {
      M << 0, 0, 0, 0;
    } else {
      M << 0, 0, 0, 0;
    }
    return M;
  }
};

template <int m, int n> class NU_XYMN_1 {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    out << sin(m * x) * sin(n * y), 0;
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::MatrixXd M(2, 2);
    M << m * cos(m * x) * sin(n * y), 0, n * cos(n * y) * sin(m * x), 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return m * cos(m * x) * sin(n * y);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
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

class NU_RADIAL_NORM_APP {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    Eigen::Vector2d out;
    out << x / r + x * x + y * y - R * R, y / r + x * x + y * y - R * R;
    if (x * x + y * y < 1.1 * 1.1 * R * R)
      return out;
    else
      return Eigen::Vector2d(x / r + x * x + y * y - 16 * R * R,
                             y / r + x * x + y * y - 16 * R * R);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r3 = r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << y * y / r3 + 2 * x, 2 * x - x * y / r3, 2 * y - x * y / r3,
        2 * y + x * x / r3;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    // if (x*x+y*y < 1.1*1.1*R*R)
    return 1 / r + 2 * x + 2 * y;
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r5 = r * r * r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << 2 - 3 * x * y * y / r5, y * (2 * x * x - y * y) / r5,
        y * (2 * x * x - y * y) / r5, 2 + x * (2 * y * y - x * x) / r5;
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = sqrt(x * x + y * y);
    double r5 = r * r * r * r * r;
    Eigen::MatrixXd M(2, 2);
    M << 2 + y * (2 * x * x - y * y) / r5, x * (2 * y * y - x * x) / r5,
        x * (2 * y * y - x * x) / r5, 2 - 3 * x * x * y / r5;
    return M;
  }
};

class G_CONST_ANN {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    /*if (x*x+y*y < (ri+ro)*(ri+ro)/4)
      return 0;
    else
      return 1;*/
    return (r - ri) / (ro - ri);
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    /*if (x*x+y*y < 4)
      return Eigen::Vector2d(0,0);
    else
      return Eigen::Vector2d(0,0);*/
    return Eigen::Vector2d(x / r, y / r) / (ro - ri);
  }
};

class NU_Y_ANN {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    Eigen::Vector2d out(0, (r - ro) / (ri - ro));
    /*if (x*x+y*y< (ri+ro)*(ri+ro)/4) {
      out << 0, 1;
    }
    else {
      out << 0,0 ;
    }*/
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    Eigen::MatrixXd M(2, 2);
    M << 0, x / r, 0, y / r;
    return M / (ri - ro);
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    return y / r / (ri - ro);
  }

  Eigen::MatrixXd dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
  Eigen::MatrixXd dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double r = std::sqrt(x * x + y * y);
    Eigen::MatrixXd M(2, 2);
    M << y * y / r / r / r, -x * y / r / r / r, -x * y / r / r / r,
        x * x / r / r / r;
    return M / (ri - ro);
  }
};

class G_poly {
public:
  double operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    double phi = atan(y / x);
    if (fabs(x) > 1.1 or fabs(y) > 1.1)
      // return 0.5*(3+2*(x*x)/(ri*ri)-1+2*y/ri);
      return 0;
    else
      return 1;
  }

  Eigen::Vector2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    // if (x*x+y*y < 4)
    //  return Eigen::Vector2d(2*x/ri/ri,1./ri);
    // else
    return Eigen::Vector2d(0, 0);
  }
};

class NU_Y_poly {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    if (fabs(x) > 1.1 or fabs(y) > 1.1) {
      out << 0, 0;
    } else {
      out << 0, 1;
    }
    return out;
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }

  double div(const Eigen::Vector2d &X) const { return 0; }

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
    Eigen::MatrixXd M(2, 2);
    M << 0, 0, 0, 0;
    return M;
  }
};

int main() {
  G g;
  // G_SIMON g;
  // G_poly g;
  // G_CONST_ANN g;
  // NU nu;
  // NU_CIRCLE nu;
  // NU_ELLIPSE nu;
  // NU_RADIAL nu;
  // NU_RADIAL_APP nu;
  // NU_RADIAL_NORM nu;
  // NU_RADIAL_NORM_APP nu;
  NU_XYMN_1<2, 3> nu;

   //std::cout << "dgrad nu: " << nu.dgrad1(Eigen::Vector2d(1.49995,-0.0124866)) << std::endl;
  //  std::cout << "expr: " << -sin(1.49995) * sin(2 * -0.0124866)
  // << std::endl;
  // NU_X nu;
  // NU_Y_ANN nu;
  // NU_Y_SIMON nu;
  // NU_Y_poly nu;
  // NU_LINEAR nu;
  // NU_CONST nu;
  /*Eigen::MatrixXd cos_list(2, 2);
  cos_list << 1, 0, 0, 0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 2, 0;*/
  // NU_CONST nu;
  /*Eigen::MatrixXd cos_list(2, 2);
  cos_list << 0.5, 0, 0, 0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 0.5, 0;
  parametricbem2d::ParametrizedFourierSum curve(cos_list, sin_list, 0,
                                                2 * M_PI);*/

  // Setting up the annular domain
  /* Eigen::MatrixXd cos_list_o(2, 2);
  cos_list_o << ro, 0, 0, 0;
  Eigen::MatrixXd sin_list_o(2, 2);
  sin_list_o << 0, 0, ro, 0;
  Eigen::MatrixXd cos_list_i(2, 2);
  cos_list_i << ri, 0, 0, 0;
  Eigen::MatrixXd sin_list_i(2, 2);
  sin_list_i << 0, 0, ri, 0;*/
  // std::cout << "LINEAR MESH!!!!" << std::endl;
  std::cout << "nu xymn1 12 , u = logr, g = logr" << std::endl;

#define SC

// kite and circle
#ifdef KNC
  std::cout << "#kite and circle" << std::endl;
  Eigen::MatrixXd cos_list_o(2, 2);
  cos_list_o << 2, 0, 0, 0;
  Eigen::MatrixXd sin_list_o(2, 2);
  sin_list_o << 0, 0, 2, 0;
  Eigen::MatrixXd cos_list_i(2, 2);
  cos_list_i << 0.25, 0.1625, 0, 0;
  Eigen::MatrixXd sin_list_i(2, 2);
  sin_list_i << 0, 0, 0.375, 0;
  parametricbem2d::ParametrizedFourierSum outer(
      Eigen::Vector2d(0, 0), cos_list_o, sin_list_o, 0, 2 * M_PI);
  parametricbem2d::ParametrizedFourierSum inner(Eigen::Vector2d(0, 0),
                                                cos_list_i, sin_list_i,
                                                2 * M_PI, 0); // inner reversed
#endif
// parametricbem2d::ParametrizedMesh mesh(curve.split(4));
// parametricbem2d::ParametrizedCircularArc
// outer(Eigen::Vector2d(0,0),R,0,2*M_PI);
// parametricbem2d::ParametrizedCircularArc
// inner(Eigen::Vector2d(R/2,0),R/4,2*M_PI,0);
// parametricbem2d::ParametrizedCircularArc
// outer(Eigen::Vector2d(0,0),4*R,0,2*M_PI);

// Concentric circle
#ifdef CONC
  std::cout << "#concentric circle" << std::endl;
  parametricbem2d::ParametrizedCircularArc outer(Eigen::Vector2d(0, 0), 4 * R,
                                                 0, 2 * M_PI);
  parametricbem2d::ParametrizedCircularArc inner(Eigen::Vector2d(0, 0), R,
                                                 2 * M_PI, 0);
#endif

// Shifted circle
#ifdef SC
  std::cout << "#Shifted circle" << std::endl;
  parametricbem2d::ParametrizedCircularArc outer(Eigen::Vector2d(2 * R, 0),
                                                 4 * R, 0, 2 * M_PI);
  parametricbem2d::ParametrizedCircularArc inner(Eigen::Vector2d(0, 0), R,
                                                 2 * M_PI, 0);
#endif

// Inner circle Outer kite
#ifdef CNK
  std::cout << "#inner circle outer kite" << std::endl;
  Eigen::MatrixXd cos_list_o(2, 2);
  cos_list_o << 3.5, 1.625, 0, 0;
  Eigen::MatrixXd sin_list_o(2, 2);
  sin_list_o << 0, 0, 3.5, 0;
  parametricbem2d::ParametrizedFourierSum outer(
      Eigen::Vector2d(0, 0), cos_list_o, sin_list_o, 0, 2 * M_PI);
  parametricbem2d::ParametrizedCircularArc inner(Eigen::Vector2d(0.8, 0.5), R,
                                                 2 * M_PI, 0);
  Eigen::MatrixXd cos_list_i(2, 2);
  cos_list_i << R, 0, 0, 0;
  Eigen::MatrixXd sin_list_i(2, 2);
  sin_list_i << 0, 0, R, 0;
  // parametricbem2d::ParametrizedFourierSum inner(Eigen::Vector2d(0.8,0.5),
  // cos_list_i, sin_list_i, 2*M_PI, 0);

#endif

#ifdef KNK
  std::cout << "#kite and kite" << std::endl;
  Eigen::MatrixXd cos_list_o(2, 2);
  cos_list_o << 3.5, 1.625, 0, 0;
  Eigen::MatrixXd sin_list_o(2, 2);
  sin_list_o << 0, 0, 3.5, 0;
  parametricbem2d::ParametrizedFourierSum outer(
      Eigen::Vector2d(0, 0), cos_list_o, sin_list_o, 0, 2 * M_PI);
  // parametricbem2d::ParametrizedFourierSum outer(cos_list_o, sin_list_o, 0, 2
  // * M_PI);
  Eigen::MatrixXd cos_list_i(2, 2);
  cos_list_i << -0.8, -0.3, 0, 0;
  // cos_list_i << 0.8, 0.3, 0, 0;
  // cos_list_i << 1, 0, 0, 0;
  Eigen::MatrixXd sin_list_i(2, 2);
  sin_list_i << 0, 0, 1, 0;
  parametricbem2d::ParametrizedFourierSum inner( // actually clockwise this way
      Eigen::Vector2d(0.5, 0.5), cos_list_i, sin_list_i, 0, 2 * M_PI);

#endif

  // Setting up a polygonal annular domain
  // The inner square
  /*Eigen::Vector2d i1(1,1);
  Eigen::Vector2d i2(-1,1);
  Eigen::Vector2d i3(-1,-1);
  Eigen::Vector2d i4(1,-1);

  // outer polygon
  Eigen::Vector2d o1(3,3);
  Eigen::Vector2d o2(1,3);
  Eigen::Vector2d o3(0,2);
  Eigen::Vector2d o4(-1,3);
  Eigen::Vector2d o5(-3,3);
  Eigen::Vector2d o6(-3,-3);
  Eigen::Vector2d o7(3,-3);

  parametricbem2d::ParametrizedLine il1(i1,i4);
  parametricbem2d::ParametrizedLine il2(i4,i3);
  parametricbem2d::ParametrizedLine il3(i3,i2);
  parametricbem2d::ParametrizedLine il4(i2,i1);

  parametricbem2d::ParametrizedLine ol1(o1,o2);
  parametricbem2d::ParametrizedLine ol2(o2,o3);
  parametricbem2d::ParametrizedLine ol3(o3,o4);
  parametricbem2d::ParametrizedLine ol4(o4,o5);
  parametricbem2d::ParametrizedLine ol5(o5,o6);
  parametricbem2d::ParametrizedLine ol6(o6,o7);
  parametricbem2d::ParametrizedLine ol7(o7,o1);

  auto add = [&](parametricbem2d::PanelVector& vector, const
  parametricbem2d::PanelVector& panels) {
    vector.insert(vector.end(),panels.begin(),panels.end());
  };

  parametricbem2d::PanelVector panels;
  add(panels,il1.split(1));
  add(panels,il2.split(1));
  add(panels,il3.split(1));
  add(panels,il4.split(1));

  add(panels,ol1.split(1));
  add(panels,ol2.split(1));
  add(panels,ol3.split(1));
  add(panels,ol4.split(1));
  add(panels,ol5.split(1));
  add(panels,ol6.split(1));
  add(panels,ol7.split(1));*/

  unsigned order = 16;
  std::cout << "#quadrature order: " << order << std::endl;
  std::cout << std::setw(10) << "#numpanels" << std::setw(25) << "c*(gradu.n)^2"
            << std::setw(25) << "BEM" << std::setw(25) << "0.5*(gradu)^2 ex."
            << std::setw(25) << "Boundary Formula 1" << std::setw(25)
            << "Boundary Formula 2" << std::endl;
  for (unsigned numpanels = 4; numpanels < 500; numpanels += 3) {
    // parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
    unsigned numpanels_i = numpanels;
    unsigned numpanels_o = numpanels;
    parametricbem2d::PanelVector panels_i = inner.split(numpanels_i);
    parametricbem2d::PanelVector panels_o = outer.split(numpanels_o);
    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(), panels_i.begin(),
                  panels_i.end()); // inner panels added first
    panels.insert(panels.end(), panels_o.begin(), panels_o.end());
    parametricbem2d::ParametrizedMesh mesh(panels);
    // parametricbem2d::ParametrizedMesh lmesh = convert_to_linear(mesh);
    double force = CalculateForce(mesh, g, nu, order);
    // Force using linear mesh!
    // double force = CalculateForce(lmesh, g, nu, order);
    // std::cout << "numpanels: " << numpanels << "  Force = " << force
    //          << std::endl;
  }

  return 0;
}
