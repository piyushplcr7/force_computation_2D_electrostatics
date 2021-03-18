#ifndef VELFIELDSHPP
#define VELFIELDSHPP

/*
 * Class representing the velocity field nu
 */
class NU_ROT {
private:
  Eigen::Vector2d center;
public:
  NU_ROT(const Eigen::Vector2d &x0):center(x0){};

  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double r = (X-center).norm();
    Eigen::Vector2d xnew = X - center;
    double x = xnew(0);
    double y = xnew(1);
    return Eigen::Vector2d(-y,x);
  }

  Eigen::MatrixXd grad(const Eigen::Vector2d &X) const {
    Eigen::MatrixXd M(2, 2);
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

#endif
