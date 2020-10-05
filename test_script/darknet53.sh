# # quantize and predict to verify
# if [ ! -f "darknet53.txt" ]; then
#   touch "darknet53.txt"
# else
#   rm "darknet53.txt"
#   touch "darknet53.txt"
# fi
# 
# # 7 bits
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m0e6" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m0e6 -qw darknet53/m0e6.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -i 2  >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m1e5" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m1e5 -qw darknet53/m1e5.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m2e4" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m2e4 -qw darknet53/m2e4.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m3e3" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m3e3 -qw darknet53/m3e3.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m4e2" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m4e2 -qw darknet53/m4e2.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m5e1" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m5e1 -qw darknet53/m5e1.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m6e0" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m6e0 -qw darknet53/m6e0.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# # 6 bits
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m0e5" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m0e5 -qw darknet53/m0e5.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -i 2  >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m1e4" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m1e4 -qw darknet53/m1e4.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m2e3" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m2e3 -qw darknet53/m2e3.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m3e2" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m3e2 -qw darknet53/m3e2.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m4e1" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m4e1 -qw darknet53/m4e1.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m5e0" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m5e0 -qw darknet53/m5e0.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# # 5 bits
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m0e4" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m0e4 -qw darknet53/m0e4.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -i 2  >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m1e3" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m1e3 -qw darknet53/m1e3.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m2e2" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m2e2 -qw darknet53/m2e2.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m3e1" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m3e1 -qw darknet53/m3e1.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m4e0" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m4e0 -qw darknet53/m4e0.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# # 4 bits
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m0e3" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m0e3 -qw darknet53/m0e3.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -i 2  >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m1e2" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m1e2 -qw darknet53/m1e2.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m2e1" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m2e1 -qw darknet53/m2e1.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# 
# echo "###############################################################" >> darknet53.txt
# echo ">> darknet53 m3e0" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/darknet53.cfg weights/darknet53.weights -t m3e0 -qw darknet53/m3e0.weights -b 1 -zm 1 >> darknet53.txt
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -nogpu >> darknet53.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -i 2 >> darknet53.txt 
# echo "" >> darknet53.txt
# echo "###############################################################" >> darknet53.txt

# validation
if [ ! -f "darknet53_valid.txt" ]; then
  touch "darknet53_valid.txt"
else
  rm "darknet53_valid.txt"
  touch "darknet53_valid.txt"
fi

# 7 bits
echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m0e6" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e6.weights -t m0e6 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m1e5" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e5.weights -t m1e5 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m2e4" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e4.weights -t m2e4 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m3e3" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e3.weights -t m3e3 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m4e2" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m4e2.weights -t m4e2 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m5e1" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m5e1.weights -t m5e1 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m6e0" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m6e0.weights -t m6e0 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

# 6 bits
echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m0e5" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e5.weights -t m0e5 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m1e4" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e4.weights -t m1e4 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m2e3" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e3.weights -t m2e3 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m3e2" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e2.weights -t m3e2 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m4e1" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m4e1.weights -t m4e1 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m5e0" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m5e0.weights -t m5e0 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

# 5 bits
echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m0e4" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e4.weights -t m0e4 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m1e3" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e3.weights -t m1e3 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m2e2" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e2.weights -t m2e2 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m3e1" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e1.weights -t m3e1 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m4e0" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m4e0.weights -t m4e0 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

# 4 bits
echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m0e3" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m0e3.weights -t m0e3 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m1e2" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m1e2.weights -t m1e2 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m2e1" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m2e1.weights -t m2e1 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt

echo "###############################################################" >> darknet53_valid.txt
echo ">> darknet53 m3e0" >> darknet53_valid.txt
echo "###############################################################" >> darknet53_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/darknet53.cfg darknet53/m3e0.weights -t m3e0 -zm 1 -i 2 >> darknet53_valid.txt 
echo "" >> darknet53_valid.txt
