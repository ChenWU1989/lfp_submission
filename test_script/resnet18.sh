# # quantize and predict to verify
# if [ ! -f "resnet18.txt" ]; then
#   touch "resnet18.txt"
# else
#   rm "resnet18.txt"
#   touch "resnet18.txt"
# fi
# 
# # 7 bits
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m0e6" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m0e6 -qw resnet18/m0e6.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -i 2  >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m1e5" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m1e5 -qw resnet18/m1e5.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m2e4" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m2e4 -qw resnet18/m2e4.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m3e3" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m3e3 -qw resnet18/m3e3.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m4e2" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m4e2 -qw resnet18/m4e2.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m5e1" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m5e1 -qw resnet18/m5e1.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m6e0" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m6e0 -qw resnet18/m6e0.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# # 6 bits
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m0e5" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m0e5 -qw resnet18/m0e5.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -i 2  >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m1e4" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m1e4 -qw resnet18/m1e4.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m2e3" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m2e3 -qw resnet18/m2e3.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m3e2" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m3e2 -qw resnet18/m3e2.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m4e1" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m4e1 -qw resnet18/m4e1.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m5e0" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m5e0 -qw resnet18/m5e0.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# # 5 bits
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m0e4" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m0e4 -qw resnet18/m0e4.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -i 2  >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m1e3" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m1e3 -qw resnet18/m1e3.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m2e2" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m2e2 -qw resnet18/m2e2.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m3e1" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m3e1 -qw resnet18/m3e1.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m4e0" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m4e0 -qw resnet18/m4e0.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# # 4 bits
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m0e3" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m0e3 -qw resnet18/m0e3.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -i 2  >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m1e2" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m1e2 -qw resnet18/m1e2.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m2e1" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m2e1 -qw resnet18/m2e1.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# 
# echo "###############################################################" >> resnet18.txt
# echo ">> resnet18 m3e0" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet18.cfg weights/resnet18.weights -t m3e0 -qw resnet18/m3e0.weights -b 1 -zm 1 >> resnet18.txt
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -nogpu >> resnet18.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -i 2 >> resnet18.txt 
# echo "" >> resnet18.txt
# echo "###############################################################" >> resnet18.txt

# validation
if [ ! -f "resnet18_valid.txt" ]; then
  touch "resnet18_valid.txt"
else
  rm "resnet18_valid.txt"
  touch "resnet18_valid.txt"
fi

# 7 bits
echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m0e6" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e6.weights -t m0e6 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m1e5" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e5.weights -t m1e5 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m2e4" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e4.weights -t m2e4 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m3e3" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e3.weights -t m3e3 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m4e2" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m4e2.weights -t m4e2 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m5e1" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m5e1.weights -t m5e1 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m6e0" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m6e0.weights -t m6e0 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

# 6 bits
echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m0e5" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e5.weights -t m0e5 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m1e4" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e4.weights -t m1e4 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m2e3" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e3.weights -t m2e3 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m3e2" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e2.weights -t m3e2 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m4e1" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m4e1.weights -t m4e1 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m5e0" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m5e0.weights -t m5e0 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

# 5 bits
echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m0e4" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e4.weights -t m0e4 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m1e3" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e3.weights -t m1e3 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m2e2" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e2.weights -t m2e2 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m3e1" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e1.weights -t m3e1 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m4e0" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m4e0.weights -t m4e0 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

# 4 bits
echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m0e3" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m0e3.weights -t m0e3 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m1e2" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m1e2.weights -t m1e2 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m2e1" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m2e1.weights -t m2e1 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt

echo "###############################################################" >> resnet18_valid.txt
echo ">> resnet18 m3e0" >> resnet18_valid.txt
echo "###############################################################" >> resnet18_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet18.cfg resnet18/m3e0.weights -t m3e0 -zm 1 -i 2 >> resnet18_valid.txt 
echo "" >> resnet18_valid.txt
