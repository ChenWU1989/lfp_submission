# # quantize and predict to verify
# if [ ! -f "alex.txt" ]; then
#   touch "alex.txt"
# else
#   rm "alex.txt"
#   touch "alex.txt"
# fi
# 
# # 7 bits
# echo "###############################################################" >> alex.txt
# echo ">> alex m0e6" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m0e6 -qw alex/m0e6.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m0e6.weights -t m0e6 -p data/cat1.jpg -zm 1 -i 2  >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m1e5" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m1e5 -qw alex/m1e5.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m1e5.weights -t m1e5 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m2e4" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m2e4 -qw alex/m2e4.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m2e4.weights -t m2e4 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m3e3" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m3e3 -qw alex/m3e3.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m3e3.weights -t m3e3 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m4e2" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m4e2 -qw alex/m4e2.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m4e2.weights -t m4e2 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m5e1" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m5e1 -qw alex/m5e1.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m5e1.weights -t m5e1 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m6e0" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m6e0 -qw alex/m6e0.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m6e0.weights -t m6e0 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# # 6 bits
# echo "###############################################################" >> alex.txt
# echo ">> alex m0e5" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m0e5 -qw alex/m0e5.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m0e5.weights -t m0e5 -p data/cat1.jpg -zm 1 -i 2  >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m1e4" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m1e4 -qw alex/m1e4.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m1e4.weights -t m1e4 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m2e3" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m2e3 -qw alex/m2e3.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m2e3.weights -t m2e3 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m3e2" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m3e2 -qw alex/m3e2.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m3e2.weights -t m3e2 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m4e1" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m4e1 -qw alex/m4e1.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m4e1.weights -t m4e1 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m5e0" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m5e0 -qw alex/m5e0.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m5e0.weights -t m5e0 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# # 5 bits
# echo "###############################################################" >> alex.txt
# echo ">> alex m0e4" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m0e4 -qw alex/m0e4.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m0e4.weights -t m0e4 -p data/cat1.jpg -zm 1 -i 2  >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m1e3" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m1e3 -qw alex/m1e3.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m1e3.weights -t m1e3 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m2e2" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m2e2 -qw alex/m2e2.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m2e2.weights -t m2e2 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m3e1" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m3e1 -qw alex/m3e1.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m3e1.weights -t m3e1 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m4e0" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m4e0 -qw alex/m4e0.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m4e0.weights -t m4e0 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# # 4 bits
# echo "###############################################################" >> alex.txt
# echo ">> alex m0e3" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m0e3 -qw alex/m0e3.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m0e3.weights -t m0e3 -p data/cat1.jpg -zm 1 -i 2  >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m1e2" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m1e2 -qw alex/m1e2.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m1e2.weights -t m1e2 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m2e1" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m2e1 -qw alex/m2e1.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m2e1.weights -t m2e1 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# 
# echo "###############################################################" >> alex.txt
# echo ">> alex m3e0" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier quantize cfg/imagenet1k.data cfg/alex.cfg weights/alex.weights -t m3e0 -qw alex/m3e0.weights -b 1 -zm 1 >> alex.txt
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -nogpu >> alex.txt 
# ./darknet q_classifier predict cfg/imagenet1k.data cfg/alex.cfg alex/m3e0.weights -t m3e0 -p data/cat1.jpg -zm 1 -i 2 >> alex.txt 
# echo "" >> alex.txt
# echo "###############################################################" >> alex.txt

# validation
if [ ! -f "alex_valid.txt" ]; then
  touch "alex_valid.txt"
else
  rm "alex_valid.txt"
  touch "alex_valid.txt"
fi

# 7 bits
echo "###############################################################" >> alex_valid.txt
echo ">> alex m0e6" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m0e6.weights -t m0e6 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m1e5" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m1e5.weights -t m1e5 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m2e4" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m2e4.weights -t m2e4 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m3e3" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m3e3.weights -t m3e3 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m4e2" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m4e2.weights -t m4e2 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m5e1" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m5e1.weights -t m5e1 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m6e0" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m6e0.weights -t m6e0 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

# 6 bits
echo "###############################################################" >> alex_valid.txt
echo ">> alex m0e5" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m0e5.weights -t m0e5 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m1e4" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m1e4.weights -t m1e4 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m2e3" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m2e3.weights -t m2e3 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m3e2" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m3e2.weights -t m3e2 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m4e1" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m4e1.weights -t m4e1 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m5e0" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m5e0.weights -t m5e0 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

# 5 bits
echo "###############################################################" >> alex_valid.txt
echo ">> alex m0e4" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m0e4.weights -t m0e4 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m1e3" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m1e3.weights -t m1e3 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m2e2" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m2e2.weights -t m2e2 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m3e1" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m3e1.weights -t m3e1 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m4e0" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m4e0.weights -t m4e0 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

# 4 bits
echo "###############################################################" >> alex_valid.txt
echo ">> alex m0e3" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m0e3.weights -t m0e3 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m1e2" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m1e2.weights -t m1e2 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m2e1" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m2e1.weights -t m2e1 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt

echo "###############################################################" >> alex_valid.txt
echo ">> alex m3e0" >> alex_valid.txt
echo "###############################################################" >> alex_valid.txt
./darknet q_classifier valid cfg/imagenet1k.data cfg/alex.cfg alex/m3e0.weights -t m3e0 -zm 1 -i 2 >> alex_valid.txt 
echo "" >> alex_valid.txt
