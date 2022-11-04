#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
//Global variables for normalization
//std::vector<double> norm_mean = {0.485, 0.456, 0.406};
//std::vector<double> norm_std = {0.229, 0.224, 0.225};
std::vector<double> norm_mean = {0.485};
std::vector<double> norm_std = {0.229};

cv::Mat crop_center(const cv::Mat &img)
{
    const int rows = img.rows;
    const int cols = img.cols;

    const int cropSize = std::min(rows,cols);
    const int offsetW = (cols - cropSize) / 2;
    const int offsetH = (rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);

    return img(roi);
}
torch::Tensor read_image(const std::string& imageName)
{
    cv::Mat img = cv::imread(imageName);
    //img = crop_center(img);
    //cv::resize(img, img, cv::Size(28,28));

    cv::imshow("image", img);

    if (img.channels()==1)
        std::cout<<"Gray scale image"<<std::endl; 
        //cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    else
        std::cout<<"BGR image"<<std::endl; 
        //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    img.convertTo(img, CV_8UC1, 1/255.0 );

    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 1}, c10::kFloat);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor.unsqueeze_(0);
   
    img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);
    
    return img_tensor.clone();
}

std::vector<std::string> load_labels(const std::string& fileName)
{
    std::ifstream ins(fileName);
    if (!ins.is_open())
    {
        std::cerr << "Couldn't open " << fileName << std::endl;
        abort();
    }

    std::vector<std::string> labels;
    std::string line;

    while (getline(ins, line))
        labels.push_back(line);

    ins.close();

    return labels;
}


int main(int argc, const char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

  std::vector<std::string> labels = load_labels(argv[2]);
  std::cout<<"labels: "<<labels<<std::endl;
 
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  torch::Tensor in = read_image(argv[3]);
  std::cout<<"input shape 0 dim "<<in.sizes()[0]<<std::endl;
  std::cout<<"input shape 1 dim "<<in.sizes()[1]<<std::endl;
  std::cout<<"input shape 2 dim"<<in.sizes()[2]<<std::endl;
  std::cout<<"input shape 3 dim"<<in.sizes()[3]<<std::endl;
  

  inputs.push_back(in);

  auto start = std::chrono::steady_clock::now();
  // Execute the model and turn its output into a tensor.
  torch::Tensor output = torch::softmax(module.forward(inputs).toTensor(), 1);
  //torch::Tensor output = module.forward(inputs).toTensor();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Elapsed time in milliSeconds : " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << std::endl;
  
  std::tuple<torch::Tensor, torch::Tensor> result = torch::max(output, 1);
  torch::Tensor prob = std::get<0>(result);
  torch::Tensor index = std::get<1>(result);

  auto probability = prob.accessor<float,1>();
  auto idx = index.accessor<long,1>();
  std::cout << " idx[0] " << idx[0] << std::endl;
  std::cout << "Predicted: " << labels[idx[0]] << std::endl;
  std::cout << "Probability: " << probability[0] << std::endl;

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';

}
