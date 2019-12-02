#include "force_calculation.hpp"
#include <iostream>
#include <functional>

int main() {
  auto f = [](double x,double y) {
    return x;
  };
  auto ll = [](double x) {
    return x-1;
  };
  auto ul = [](double x) {
    return 1-x;
  };
  double integral = ComputeDoubleIntegral(f,-1,1,ll,ul,10);
  std::cout << "double integral = " << integral << std::endl;
  return 0;
}
