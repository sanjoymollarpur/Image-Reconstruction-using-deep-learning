## Deep Learning Optoacoustic Tomography with Sparse Data
This repository contains a TensorFlow implementation of the paper titled "Deep Learning Optoacoustic Tomography with Sparse Data." The original implementation was provided in Theano by the paper's authors. In this project, we have translated their work to TensorFlow and conducted a comparative analysis.

#Overview
Optoacoustic tomography is a powerful imaging technique, and this paper introduces a deep learning-based approach to enhance reconstruction accuracy with sparse data. Our contribution involves reimplementing the authors' model in TensorFlow, enabling broader community access and ease of integration into existing TensorFlow-based projects.


#Key Features

TensorFlow implementation of the deep learning model proposed in the paper.
Utilized the provided test dataset with 10 samples for evaluation.
Conducted predictions using both the authors' pre-trained model and our implemented model.
Calculated the Structural Similarity Index (SSIM) matrix for quantitative comparison.

#Results
SSIM Matrix for Authors' Pretrained Model: 0.63
SSIM Matrix for Our Implemented Model: 0.68


![0](https://github.com/sanjoymollarpur/Image-Reconstruction-using-deep-learning/assets/89268947/313f984a-e3d6-4f1b-a667-49bb3b99adf9)

git clone https://github.com/yourusername/your-repo.git
cd your-repo

pip install -r requirements.txt

