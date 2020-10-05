
#./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg-16.cfg weights/vgg-16.weights -t fp6_m5e0 -qw vgg_m5e0.weights -b 1 -zm 1

# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg-16.cfg vgg_m5e0.weights -t fp6_m5e0 -p data/cat1.jpg -zm 1

# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t fp8_m7e0 -qw resnet18_m7e0.weights -b 1 -zm 1

# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18_m7e0.weights -t fp8_m7e0 -p data/cat1.jpg -zm 1

./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m3e2 -qw resnet18_m3e2.weights -b 1 -zm 1

./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18_m3e2.weights -t m3e2 -p data/cat1.jpg -nogpu -zm 1
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18_m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18_m3e2.weights -t m3e2
