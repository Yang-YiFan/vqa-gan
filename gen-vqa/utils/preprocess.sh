echo 'Downloading Datasets...'
./download.sh
echo 'Organizing Datasets Folder...'
./organize_datasets.sh
echo 'Resizing Images...'
python3 resize_images.py --input_dir='../datasets/Images' --output_dir='../datasets/Resized_Images'
echo 'Processing Texts...'
python3 make_vacabs_for_questions_answers.py --input_dir='../datasets'
echo 'Building VQA Inputs...'
python3 build_vqa_inputs.py --input_dir='../datasets' --output_dir='../datasets'
echo 'Preprocessing Finished'
