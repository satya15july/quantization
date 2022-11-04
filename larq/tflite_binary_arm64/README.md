##Build Instruction for arm64(raspi-4)
1.  Clone [Larq Compute Engine](https://github.com/larq/compute-engine). 
2.  Copy the label_image from "tensorflow/tensorflow/lite/examples/label_image" & put inside root directory of compute-engine.
3.  Replace BUILD file present inside label_image with BUILD file present inside tflite_binary_arm64.
4.  Build with bazel:
    -   bazel build --config=aarch64 //label_image:label_image

##Run command

./label_image --tflite_model /tmp/mobilenet_v1_1.0_224.tflite --labels labels.txt --image <image_path> --lcompute 1

Use *"--lcompute 1"* when Binarized neural network is in use.Otherwise, use *"--lcompute 0"*.




