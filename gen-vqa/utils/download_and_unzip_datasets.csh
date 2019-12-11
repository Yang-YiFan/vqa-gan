#!/bin/tcsh

#########################################################

# One may need to change directory for datasets like this.
#set DATASETS_DIR = "/run/media/hoosiki/WareHouse3/mtb/datasets/VQA"

mkdir -p "../datasets"
set DATASETS_DIR = "../datasets"

##########################################################

set ANNOTATIONS_DIR = "${DATASETS_DIR}/Annotations"
set QUESTIONS_DIR = "${DATASETS_DIR}/Questions"
set IMAGES_DIR = "${DATASETS_DIR}/Images"

##########################################################

mkdir -p ${ANNOTATIONS_DIR}
mkdir -p ${QUESTIONS_DIR}
mkdir -p ${IMAGES_DIR}

##########################################################

# Download datasets from VQA official url: https://visualqa.org/download.html

# VQA Annotations
wget -O ${ANNOTATIONS_DIR}/v2_Annotations_Train_abstract.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Train_abstract_v002.zip"
wget -O ${ANNOTATIONS_DIR}/v2_Annotations_Val_abstract.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Annotations_Val_abstract_v002.zip"

# VQA Input Questions
wget -O ${QUESTIONS_DIR}/v2_Questions_Train_abstract.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Train_abstract_v002.zip"
wget -O ${QUESTIONS_DIR}/v2_Questions_Val_abstract.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Val_abstract_v002.zip"
wget -O ${QUESTIONS_DIR}/v2_Questions_Test_abstract.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/vqa/Questions_Test_abstract_v002.zip"

# VQA Input Images (abstract)
wget -O ${IMAGES_DIR}/train2014.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_train2015.zip"
wget -O ${IMAGES_DIR}/val2014.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_val2015.zip"
wget -O ${IMAGES_DIR}/test2015.zip "https://s3.amazonaws.com/cvmlp/vqa/abstract_v002/scene_img/scene_img_abstract_v002_test2015.zip"

##########################################################

unzip ${ANNOTATIONS_DIR}/v2_Annotations_Train_abstract.zip -d ${ANNOTATIONS_DIR}
unzip ${ANNOTATIONS_DIR}/v2_Annotations_Val_abstract.zip -d ${ANNOTATIONS_DIR}

unzip ${QUESTIONS_DIR}/v2_Questions_Train_abstract.zip -d ${QUESTIONS_DIR}
unzip ${QUESTIONS_DIR}/v2_Questions_Val_abstract.zip -d ${QUESTIONS_DIR}
unzip ${QUESTIONS_DIR}/v2_Questions_Test_abstract.zip -d ${QUESTIONS_DIR}

unzip ${IMAGES_DIR}/train2014.zip -d ${IMAGES_DIR}
unzip ${IMAGES_DIR}/val2014.zip -d ${IMAGES_DIR}
unzip ${IMAGES_DIR}/test2015.zip -d ${IMAGES_DIR}

##########################################################

# Remove unnecessary zip files.

rm ${ANNOTATIONS_DIR}/v2_Annotations_Train_abstract.zip
rm ${ANNOTATIONS_DIR}/v2_Annotations_Val_abstract.zip

rm ${QUESTIONS_DIR}/v2_Questions_Train_abstract.zip
rm ${QUESTIONS_DIR}/v2_Questions_Val_abstract.zip
rm ${QUESTIONS_DIR}/v2_Questions_Test_abstract.zip

rm ${IMAGES_DIR}/train2014.zip
rm ${IMAGES_DIR}/val2014.zip
rm ${IMAGES_DIR}/test2015.zip

##########################################################
