#include <torch/torch.h>
#include <iostream>

std::tuple<torch::Tensor, torch::Tensor>  create_dummy_data(void);

struct Net : torch::nn::Module{
    Net(int in_dim, int out_dim) {
        linear_layer = register_module("fc1", torch::nn::Linear(in_dim, out_dim));
    }
    torch::Tensor forward(torch::Tensor x){
        x = linear_layer->forward(x);
        return x;
    }
    torch::nn::Linear linear_layer{ nullptr };
};


int main(){
    
    auto data = create_dummy_data();
    auto inputs = std::get<0>(data);
    auto targets = std::get<1>(data);
    
    int inputDim = 5;
    int outputDim = 5;
    
    auto net = std::make_shared<Net>(inputDim,outputDim);

    const size_t epoch_size = 100;
    const float learning_rate = 0.001;

    torch::optim::SGD optimizer(net->parameters(), learning_rate);

    
    for(size_t epoch = 1; epoch <= epoch_size; epoch ++)
    {
        torch::Tensor prediction = net->forward(inputs);
        optimizer.zero_grad();

        auto loss = torch::mse_loss(prediction, targets);
        loss.backward();
        optimizer.step();

        float loss_val = loss.item<float>();
        std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>() << std::endl;

        torch::save(net, "net.pt");
        
    }


    return 0;
}

std::tuple<torch::Tensor, torch::Tensor>  create_dummy_data(void){

    torch::Tensor input = torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0});
    torch::Tensor target = torch::tensor({3.0, 5.0, 7.0, 9.0, 11.0});

    return std::make_tuple(input,target);
}