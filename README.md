# intelunnati_Duality
Title: Conquering Fashion MNIST with CNNs using Computer Vision
Team Name: Duality
Authors: DANIEL PREM
         DHARSHAN DELWIN D

Date of Submission: 14/07/2023

**Abstract**:

Introduction:
The classification of clothing items plays a crucial role in various applications, such as e-commerce, fashion industry analysis, and personalized recommendation systems. The Fashion MNIST dataset, consisting of grayscale images of fashion products, provides a suitable benchmark for developing clothing classification models. In this project, our aim is to design and train a CNN model capable of accurately categorizing the images into ten clothing classes.

**Motivation behind the Problem:**
Accurate clothing classification has numerous practical applications and benefits. In the e-commerce industry, it enables automated product categorization and personalized shopping experiences. In the fashion industry, it aids in trend analysis and inventory management. Moreover, efficient and accurate clothing classification is essential for recommendation systems, allowing users to discover relevant fashion items based on their preferences. By developing an efficient CNN model, we aim to contribute to the field of computer vision and empower various industries with advanced image classification capabilities.
Prior Work/Background:
In recent years, significant advancements have been made in image classification and deep learning techniques. Several studies have focused on the Fashion MNIST dataset to develop clothing classification models. For instance, researchers have explored different CNN architectures, such as LeNet-5, VGG, and ResNet, to achieve high accuracy in classifying fashion images. These studies have demonstrated the effectiveness of CNNs in capturing intricate features and patterns in clothing images.
Additionally, optimization techniques have been applied to improve model performance and efficiency. Intel provides optimization tools and libraries specifically designed to enhance deep learning models on Intel architectures. Notable optimizations include the Intel optimization tool OPENVINO . This tool leverage the hardware capabilities and optimizations offered by Intel processors, resulting in improved inference speed and efficiency.

**Our Approach:**
In this project, we followed a systematic approach to develop an efficient clothing classification model using CNNs and Intel optimization. The key steps in our approach were as follows:
•	Dataset Preprocessing: We performed data preprocessing, including scaling the pixel values, reshaping the data, and one-hot encoding the labels.
•	CNN Model Architecture: We designed and implemented three different CNN models. Model 1 consisted of multiple convolutional and pooling layers, followed by fully connected layers. Model 2 incorporated ReduceLROnPlateau callback for learning rate reduction during training. Model 3 utilized the VGG architecture, known for its depth and performance in image classification tasks.
•	Model Training: We split the dataset into training and testing sets, and the models were trained using the training set. We employed appropriate loss functions, optimizers, and evaluation metrics to train and evaluate the models.
•	Intel Optimization: We explored Intel optimization tool OPENVINO to accelerate the inference speed of our trained models. We evaluated the performance improvements achieved through Intel optimization.

**Results:**
The performance of our developed models and the impact of Intel optimization are as follows:
Model 1: The efficient CNN model achieved an accuracy of 92.25% on the test set. It demonstrated effective feature extraction and classification capabilities.
 
Model 2: The CNN model with ReduceLROnPlateau callback achieved a similar accuracy of 92.06% on the test set. The learning rate reduction strategy helped improve model convergence and generalization.
 
Model 3 :The VGG-based CNN model achieved an accuracy of 91.5% on the test set. The deeper architecture provided a better understanding of complex fashion patterns.

Intel Optimization(OPENVINO): By utilizing Intel optimization tool OPENVINO we observed a noticeable improvement in inference speed compared to the TensorFlow implementation.
 
 
**References:**
Xiao, H., et al. (2017). Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv preprint arXiv:1708.07747. Retrieved from https://arxiv.org/abs/1708.07747
Intel Corporation. (n.d.). OpenVINO™ Toolkit Documentation. Retrieved from https://docs.openvinotoolkit.org/
Intel Corporation. (n.d.). OpenVINO™ Model Optimizer Developer Guide. Retrieved from https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html



