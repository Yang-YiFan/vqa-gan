#########################################################

# One may need to change directory for datasets like this.
#set DATASETS_DIR = "/run/media/hoosiki/WareHouse3/mtb/datasets/VQA"

mkdir -p "../datasets"

mkdir -p "../datasets/Annotations"
mkdir -p "../datasets/Questions"
mkdir -p "../datasets/Images"

##########################################################

# Download datasets from VQA official url: https://visualqa.org/download.html

# VQA Annotations
wget -O ../datasets/Annotations/v2_Annotations_Train_abstract.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Train_abstract_v002.zip"
wget -O ../datasets/Annotations/v2_Annotations_Val_abstract.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Val_abstract_v002.zip"

# VQA Input Questions
wget -O ../datasets/Questions/v2_Questions_Train_abstract.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Train_abstract_v002.zip"
wget -O ../datasets/Questions/v2_Questions_Val_abstract.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Val_abstract_v002.zip"
wget -O ../datasets/Questions/v2_Questions_Test_abstract.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Test_abstract_v002.zip"

# VQA Input Images (abstract)
wget -O ../datasets/Images/train2014.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_train2015.zip"
wget -O ../datasets/Images/val2014.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_val2015.zip"
wget -O ../datasets/Images/test2015.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_test2015.zip"

##########################################################

unzip ../datasets/Annotations/v2_Annotations_Train_abstract.zip -d ../datasets/Annotations
unzip ../datasets/Annotations/v2_Annotations_Val_abstract.zip -d ../datasets/Annotations

unzip ../datasets/Questions/v2_Questions_Train_abstract.zip -d ../datasets/Questions
unzip ../datasets/Questions/v2_Questions_Val_abstract.zip -d ../datasets/Questions
unzip ../datasets/Questions/v2_Questions_Test_abstract.zip -d ../datasets/Questions

unzip ../datasets/Images/train2014.zip -d ../datasets/Images
unzip ../datasets/Images/val2014.zip -d ../datasets/Images
unzip ../datasets/Images/test2015.zip -d ../datasets/Images

##########################################################

# Remove unnecessary zip files.

rm ../datasets/Annotations/v2_Annotations_Train_abstract.zip
rm ../datasets/Annotations/v2_Annotations_Val_abstract.zip

rm ../datasets/Questions/v2_Questions_Train_abstract.zip
rm ../datasets/Questions/v2_Questions_Val_abstract.zip
rm ../datasets/Questions/v2_Questions_Test_abstract.zip

rm ../datasets/Images/train2014.zip
rm ../datasets/Images/val2014.zip
rm ../datasets/Images/test2015.zip

##########################################################
