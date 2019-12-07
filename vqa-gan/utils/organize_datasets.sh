mkdir ../datasets/Images/train2015
mkdir ../datasets/Images/val2015
mkdir ../datasets/Images/test2015
mv ../datasets/Images/*train2015_* ../datasets/Images/train2015/
mv ../datasets/Images/*val2015_*   ../datasets/Images/val2015/
mv ../datasets/Images/*test2015_*  ../datasets/Images/test2015/

rm ../datasets/Annotations/*.zip
rm ../datasets/Questions/*.zip
