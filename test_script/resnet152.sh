# quantize and predict to verify
if [ ! -f "resnet152.txt" ]; then
  touch "resnet152.txt"
else
  rm "resnet152.txt"
  touch "resnet152.txt"
fi

# 8 bits
echo "###############################################################" >> resnet152.txt
echo ">> resnet152 m0e7" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m0e7 -qw resnet152/m0e7.weights -b 1 -zm 1 >> resnet152.txt
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e7.weights -t m0e7 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e7.weights -t m0e7 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt

echo "###############################################################" >> resnet152.txt
echo ">> resnet152 m1e6" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m1e6 -qw resnet152/m1e6.weights -b 1 -zm 1 >> resnet152.txt
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e6.weights -t m1e6 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e6.weights -t m1e6 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt

echo "###############################################################" >> resnet152.txt
echo ">> resnet152 m2e5" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m2e5 -qw resnet152/m2e5.weights -b 1 -zm 1 >> resnet152.txt
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e5.weights -t m2e5 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e5.weights -t m2e5 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt

echo "###############################################################" >> resnet152.txt
echo ">> resnet152 m3e4" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m3e4 -qw resnet152/m3e4.weights -b 1 -zm 1 >> resnet152.txt
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e4.weights -t m3e4 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e4.weights -t m3e4 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt

echo "###############################################################" >> resnet152.txt
echo ">> resnet152 m4e3" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m4e3 -qw resnet152/m4e3.weights -b 1 -zm 1 >> resnet152.txt
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e3.weights -t m4e3 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e3.weights -t m4e3 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt

echo "###############################################################" >> resnet152.txt
echo ">> resnet152 m5e2" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m5e2 -qw resnet152/m5e2.weights -b 1 -zm 1 >> resnet152.txt
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m5e2.weights -t m5e2 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m5e2.weights -t m5e2 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt

echo "###############################################################" >> resnet152.txt
echo ">> resnet152 m6e1" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m6e1 -qw resnet152/m6e1.weights -b 1 -zm 1 >> resnet152.txt
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m6e1.weights -t m6e1 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m6e1.weights -t m6e1 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt

echo "###############################################################" >> resnet152.txt
echo ">> resnet152 m7e0" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m7e0 -qw resnet152/m7e0.weights -b 1 -zm 1 >> resnet152.txt
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m7e0.weights -t m7e0 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m7e0.weights -t m7e0 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
echo "" >> resnet152.txt
echo "###############################################################" >> resnet152.txt

# # 7 bits
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m0e6" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m0e6 -qw resnet152/m0e6.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m1e5" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m1e5 -qw resnet152/m1e5.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m2e4" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m2e4 -qw resnet152/m2e4.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m3e3" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m3e3 -qw resnet152/m3e3.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m4e2" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m4e2 -qw resnet152/m4e2.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m5e1" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m5e1 -qw resnet152/m5e1.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m6e0" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m6e0 -qw resnet152/m6e0.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# # 6 bits
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m0e5" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m0e5 -qw resnet152/m0e5.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m1e4" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m1e4 -qw resnet152/m1e4.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m2e3" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m2e3 -qw resnet152/m2e3.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m3e2" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m3e2 -qw resnet152/m3e2.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m4e1" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m4e1 -qw resnet152/m4e1.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m5e0" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m5e0 -qw resnet152/m5e0.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# # 5 bits
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m0e4" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m0e4 -qw resnet152/m0e4.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m1e3" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m1e3 -qw resnet152/m1e3.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m2e2" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m2e2 -qw resnet152/m2e2.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m3e1" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m3e1 -qw resnet152/m3e1.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m4e0" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m4e0 -qw resnet152/m4e0.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# # 4 bits
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m0e3" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m0e3 -qw resnet152/m0e3.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -i 2  >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m1e2" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m1e2 -qw resnet152/m1e2.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m2e1" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m2e1 -qw resnet152/m2e1.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# echo "###############################################################" >> resnet152.txt
# echo ">> resnet152 m3e0" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/resnet152.cfg weights/resnet152.weights -t m3e0 -qw resnet152/m3e0.weights -b 1 -zm 1 >> resnet152.txt
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -nogpu >> resnet152.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -i 2 >> resnet152.txt 
# echo "" >> resnet152.txt
# echo "###############################################################" >> resnet152.txt
# 
# # validation
# if [ ! -f "resnet152_valid.txt" ]; then
#   touch "resnet152_valid.txt"
# else
#   rm "resnet152_valid.txt"
#   touch "resnet152_valid.txt"
# fi

# 8 bits
echo "###############################################################" >> resnet152_valid.txt
echo ">> resnet152 m0e7" >> resnet152_valid.txt
echo "###############################################################" >> resnet152_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e7.weights -t m0e7 -zm 1 -i 2 >> resnet152_valid.txt 
echo "" >> resnet152_valid.txt

echo "###############################################################" >> resnet152_valid.txt
echo ">> resnet152 m1e6" >> resnet152_valid.txt
echo "###############################################################" >> resnet152_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e6.weights -t m1e6 -zm 1 -i 2 >> resnet152_valid.txt 
echo "" >> resnet152_valid.txt

echo "###############################################################" >> resnet152_valid.txt
echo ">> resnet152 m2e5" >> resnet152_valid.txt
echo "###############################################################" >> resnet152_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e5.weights -t m2e5 -zm 1 -i 2 >> resnet152_valid.txt 
echo "" >> resnet152_valid.txt

echo "###############################################################" >> resnet152_valid.txt
echo ">> resnet152 m3e4" >> resnet152_valid.txt
echo "###############################################################" >> resnet152_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e4.weights -t m3e4 -zm 1 -i 2 >> resnet152_valid.txt 
echo "" >> resnet152_valid.txt

echo "###############################################################" >> resnet152_valid.txt
echo ">> resnet152 m4e3" >> resnet152_valid.txt
echo "###############################################################" >> resnet152_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e3.weights -t m4e3 -zm 1 -i 2 >> resnet152_valid.txt 
echo "" >> resnet152_valid.txt

echo "###############################################################" >> resnet152_valid.txt
echo ">> resnet152 m5e2" >> resnet152_valid.txt
echo "###############################################################" >> resnet152_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m5e2.weights -t m5e2 -zm 1 -i 2 >> resnet152_valid.txt 
echo "" >> resnet152_valid.txt

echo "###############################################################" >> resnet152_valid.txt
echo ">> resnet152 m6e1" >> resnet152_valid.txt
echo "###############################################################" >> resnet152_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m6e1.weights -t m6e1 -zm 1 -i 2 >> resnet152_valid.txt 
echo "" >> resnet152_valid.txt

echo "###############################################################" >> resnet152_valid.txt
echo ">> resnet152 m7e0" >> resnet152_valid.txt
echo "###############################################################" >> resnet152_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m7e0.weights -t m7e0 -zm 1 -i 2 >> resnet152_valid.txt 
echo "" >> resnet152_valid.txt

# # 7 bits
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m0e6" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e6.weights -t m0e6 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m1e5" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e5.weights -t m1e5 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m2e4" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e4.weights -t m2e4 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m3e3" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e3.weights -t m3e3 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m4e2" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e2.weights -t m4e2 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m5e1" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m5e1.weights -t m5e1 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m6e0" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m6e0.weights -t m6e0 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# # 6 bits
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m0e5" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e5.weights -t m0e5 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m1e4" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e4.weights -t m1e4 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m2e3" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e3.weights -t m2e3 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m3e2" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e2.weights -t m3e2 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m4e1" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e1.weights -t m4e1 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m5e0" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m5e0.weights -t m5e0 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# # 5 bits
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m0e4" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e4.weights -t m0e4 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m1e3" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e3.weights -t m1e3 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m2e2" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e2.weights -t m2e2 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m3e1" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e1.weights -t m3e1 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m4e0" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m4e0.weights -t m4e0 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# # 4 bits
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m0e3" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m0e3.weights -t m0e3 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m1e2" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m1e2.weights -t m1e2 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m2e1" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m2e1.weights -t m2e1 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
# 
# echo "###############################################################" >> resnet152_valid.txt
# echo ">> resnet152 m3e0" >> resnet152_valid.txt
# echo "###############################################################" >> resnet152_valid.txt
# ./darknet q_classifier valid cfg/imagenet1k.data cfg/resnet152.cfg resnet152/m3e0.weights -t m3e0 -zm 1 -i 2 >> resnet152_valid.txt 
# echo "" >> resnet152_valid.txt
