#include <torch/torch.h>
#include <iostream>


int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  
  std::cout<<"Hello Torch! this is my first tensor"<<std::endl;
  std::cout << tensor << std::endl;

  
}