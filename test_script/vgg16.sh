# # quantize and predict to verify
# if [ ! -f "vgg16.txt" ]; then
#   touch "vgg16.txt"
# else
#   rm "vgg16.txt"
#   touch "vgg16.txt"
# fi
# 
# # 7 bits
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m0e6" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m0e6 -qw vgg16/m0e6.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -i 2  >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m1e5" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m1e5 -qw vgg16/m1e5.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m2e4" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m2e4 -qw vgg16/m2e4.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m3e3" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m3e3 -qw vgg16/m3e3.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m4e2" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m4e2 -qw vgg16/m4e2.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m5e1" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m5e1 -qw vgg16/m5e1.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m6e0" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m6e0 -qw vgg16/m6e0.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# # 6 bits
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m0e5" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m0e5 -qw vgg16/m0e5.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -i 2  >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m1e4" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m1e4 -qw vgg16/m1e4.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m2e3" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m2e3 -qw vgg16/m2e3.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m3e2" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m3e2 -qw vgg16/m3e2.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m4e1" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m4e1 -qw vgg16/m4e1.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m5e0" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m5e0 -qw vgg16/m5e0.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# # 5 bits
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m0e4" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m0e4 -qw vgg16/m0e4.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -i 2  >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m1e3" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m1e3 -qw vgg16/m1e3.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m2e2" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m2e2 -qw vgg16/m2e2.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m3e1" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m3e1 -qw vgg16/m3e1.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m4e0" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m4e0 -qw vgg16/m4e0.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# # 4 bits
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m0e3" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m0e3 -qw vgg16/m0e3.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -i 2  >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m1e2" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m1e2 -qw vgg16/m1e2.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m2e1" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m2e1 -qw vgg16/m2e1.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# 
# echo "###############################################################" >> vgg16.txt
# echo ">> vgg16 m3e0" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/vgg16.cfg weights/vgg16.weights -t m3e0 -qw vgg16/m3e0.weights -b 1 -zm 1 >> vgg16.txt
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -nogpu >> vgg16.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -i 2 >> vgg16.txt 
# echo "" >> vgg16.txt
# echo "###############################################################" >> vgg16.txt

# validation
if [ ! -f "vgg16_valid.txt" ]; then
  touch "vgg16_valid.txt"
else
  rm "vgg16_valid.txt"
  touch "vgg16_valid.txt"
fi

# 7 bits
echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m0e6" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e6.weights -t m0e6 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m1e5" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e5.weights -t m1e5 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m2e4" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e4.weights -t m2e4 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m3e3" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e3.weights -t m3e3 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m4e2" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m4e2.weights -t m4e2 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m5e1" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m5e1.weights -t m5e1 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m6e0" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m6e0.weights -t m6e0 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

# 6 bits
echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m0e5" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e5.weights -t m0e5 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m1e4" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e4.weights -t m1e4 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m2e3" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e3.weights -t m2e3 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m3e2" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e2.weights -t m3e2 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m4e1" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m4e1.weights -t m4e1 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m5e0" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m5e0.weights -t m5e0 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

# 5 bits
echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m0e4" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e4.weights -t m0e4 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m1e3" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e3.weights -t m1e3 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m2e2" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e2.weights -t m2e2 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m3e1" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e1.weights -t m3e1 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m4e0" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m4e0.weights -t m4e0 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

# 4 bits
echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m0e3" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m0e3.weights -t m0e3 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m1e2" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m1e2.weights -t m1e2 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m2e1" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m2e1.weights -t m2e1 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt

echo "###############################################################" >> vgg16_valid.txt
echo ">> vgg16 m3e0" >> vgg16_valid.txt
echo "###############################################################" >> vgg16_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/vgg16.cfg vgg16/m3e0.weights -t m3e0 -zm 1 -i 2 >> vgg16_valid.txt 
echo "" >> vgg16_valid.txt
