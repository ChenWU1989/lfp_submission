# # quantize and predict to verify
# if [ ! -f "densenet201.txt" ]; then
#   touch "densenet201.txt"
# else
#   rm "densenet201.txt"
#   touch "densenet201.txt"
# fi
# 
# # 7 bits
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m0e6" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m0e6 -qw densenet201/m0e6.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -i 2  >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m1e5" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m1e5 -qw densenet201/m1e5.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m2e4" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m2e4 -qw densenet201/m2e4.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m3e3" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m3e3 -qw densenet201/m3e3.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m4e2" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m4e2 -qw densenet201/m4e2.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m5e1" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m5e1 -qw densenet201/m5e1.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m6e0" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m6e0 -qw densenet201/m6e0.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# # 6 bits
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m0e5" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m0e5 -qw densenet201/m0e5.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -i 2  >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m1e4" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m1e4 -qw densenet201/m1e4.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m2e3" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m2e3 -qw densenet201/m2e3.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m3e2" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m3e2 -qw densenet201/m3e2.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m4e1" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m4e1 -qw densenet201/m4e1.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m5e0" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m5e0 -qw densenet201/m5e0.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# # 5 bits
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m0e4" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m0e4 -qw densenet201/m0e4.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -i 2  >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m1e3" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m1e3 -qw densenet201/m1e3.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m2e2" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m2e2 -qw densenet201/m2e2.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m3e1" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m3e1 -qw densenet201/m3e1.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m4e0" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m4e0 -qw densenet201/m4e0.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# # 4 bits
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m0e3" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m0e3 -qw densenet201/m0e3.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -i 2  >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m1e2" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m1e2 -qw densenet201/m1e2.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m2e1" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m2e1 -qw densenet201/m2e1.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# 
# echo "###############################################################" >> densenet201.txt
# echo ">> densenet201 m3e0" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/densenet201.cfg weights/densenet201.weights -t m3e0 -qw densenet201/m3e0.weights -b 1 -zm 1 >> densenet201.txt
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -nogpu >> densenet201.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -i 2 >> densenet201.txt 
# echo "" >> densenet201.txt
# echo "###############################################################" >> densenet201.txt

# validation
if [ ! -f "densenet201_valid.txt" ]; then
  touch "densenet201_valid.txt"
else
  rm "densenet201_valid.txt"
  touch "densenet201_valid.txt"
fi

# 7 bits
echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m0e6" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e6.weights -t m0e6 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m1e5" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e5.weights -t m1e5 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m2e4" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e4.weights -t m2e4 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m3e3" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e3.weights -t m3e3 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m4e2" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m4e2.weights -t m4e2 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m5e1" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m5e1.weights -t m5e1 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m6e0" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m6e0.weights -t m6e0 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

# 6 bits
echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m0e5" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e5.weights -t m0e5 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m1e4" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e4.weights -t m1e4 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m2e3" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e3.weights -t m2e3 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m3e2" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e2.weights -t m3e2 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m4e1" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m4e1.weights -t m4e1 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m5e0" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m5e0.weights -t m5e0 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

# 5 bits
echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m0e4" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e4.weights -t m0e4 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m1e3" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e3.weights -t m1e3 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m2e2" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e2.weights -t m2e2 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m3e1" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e1.weights -t m3e1 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m4e0" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m4e0.weights -t m4e0 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

# 4 bits
echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m0e3" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m0e3.weights -t m0e3 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m1e2" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m1e2.weights -t m1e2 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m2e1" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m2e1.weights -t m2e1 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt

echo "###############################################################" >> densenet201_valid.txt
echo ">> densenet201 m3e0" >> densenet201_valid.txt
echo "###############################################################" >> densenet201_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/densenet201.cfg densenet201/m3e0.weights -t m3e0 -zm 1 -i 2 >> densenet201_valid.txt 
echo "" >> densenet201_valid.txt
