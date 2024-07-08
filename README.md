# Gender Classification with Transfer Learning

## Background
The development of a photo classification application based on gender has become important in various usage contexts, such as demographic analysis for social surveys and market research, effective advertising and marketing by targeting gender-specific markets, personalized content delivery on social media platforms, and contributing to public safety and crime handling through gender identification from images or video recordings. Additionally, this application can support the development of products tailored to gender preferences. Despite its benefits, it is important to remember that the use of personal data, including gender identification, must comply with strict privacy regulations, and this application requires careful testing to reduce unwanted bias and misclassification while considering ethical and privacy issues.

## Objectives and Goals
The purpose of creating a photo classification application based on gender is to identify and separate photos based on the gender (male or female) of the subjects in the images. This can be used for various purposes, such as demographic analysis, market targeting, personalized user experience, and crime security. This application can help in collecting important demographic data, presenting more relevant advertisements, providing personalized content, supporting crime handling, and developing products that align with specific gender preferences.

## Methodology
### Data Understanding
CelebA (Celebrities Attributes) is a dataset containing numerous faces of public figures. This dataset is often used for tasks in the field of computer vision, such as face detection, face recognition, emotion recognition, facial attribute analysis, and more.

![CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA/intro.png)

Here are some general information about the CelebA dataset:

* **Data Amount:** The CelebA dataset consists of over 200,000 celebrity face images.
* **Annotations:** Each image in this dataset comes with several attribute annotations, such as gender, presence of a smile, hair type, and more. These annotations are used to train machine learning models to recognize and classify facial attributes in images.
* **Format:** The images in this dataset vary in size but are generally $178\times218$ pixels. The common file format used is JPEG.

This dataset is widely used by researchers and practitioners in the computer vision community to develop and evaluate various models. Additionally, there are challenges associated with this dataset due to the significant variation in face positions, expressions, and lighting, which can help in developing robust and outlier-resistant models.

### Data Preparation
Out of the total 200,000 data points in the CelebA dataset, 5,017 were provided to the Indonesia AI team for processing. After scrutiny, 17 duplicates were found, leaving a total of 5,000 images used. This experiment does not focus on the detailed implementation steps but rather provides a brief overview of transfer learning usage. Therefore, not much preprocessing was done in this experiment, except ensuring there were no duplicate data as mentioned above. Other data exploration conducted includes examining a bar plot of male and female labels.

![Data Distribution](https://github.com/emhaihsan/gender-classification/blob/main/img/grafik.png)

It can be seen that data with label 0 (female) has more data compared to data with label 1 (male).

### Modeling
In this experiment, transfer learning was attempted using the following three pretrained architectures:
#### 1. Inception V3:
Inception V3 is a convolutional neural network model developed by Google's team. This model uses complex "Inception" blocks for computational efficiency and addresses the vanishing gradients problem.

![Inception](https://production-media.paperswithcode.com/methods/inceptionv3onc--oview_vjAbOfw.png)

Reference Paper:
*"Rethinking the Inception Architecture for Computer Vision"*
Authors: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna

[Paper link](https://arxiv.org/abs/1512.00567v3)

#### 2. ResNet (Residual Network):
ResNet is a convolutional neural network model proposed by Kaiming He and his colleagues. This model uses "residual" blocks to tackle the vanishing gradient problem when training very deep networks.

![ResNet](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*9LqUp7XyEx1QNc6A.png)

Reference Paper:
*"Deep Residual Learning for Image Recognition"*
Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

[Paper link](https://arxiv.org/abs/1512.03385)

#### 3. Xception:
Xception is a convolutional neural network model that develops the idea from Inception V3. It uses "depthwise separable convolution" to replace traditional convolution, reducing the number of parameters and increasing efficiency.

![Xception](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*hOcAEj9QzqgBXcwUzmEvSg.png)

Reference Paper:
*"Xception: Deep Learning with Depthwise Separable Convolutions"*
Authors: François Chollet

[Paper link](https://arxiv.org/abs/1610.02357)

The transfer learning implementation using the three algorithms can be seen in the links provided in the table below:
| No  | Model       | Fine-Tuning at Layer | Link |
|-----|-------------|----------------------|------|
| 1   | InceptionV3 | Last 40 layers       | [Inception](https://github.com/emhaihsan/gender-classification/blob/main/inception.ipynb) |
| 2   | Resnet50V2  | Last 52 layers       | [Resnet](https://github.com/emhaihsan/gender-classification/blob/main/resnet.ipynb) |
| 3   | Xception    | Last 30 layers       | [Xception](https://github.com/emhaihsan/gender-classification/blob/main/xception.ipynb) |

In addition to fine-tuning, all three notebooks were trained using identical parameters. The parameters used are as follows:
* **Image Size:** $218 \times 178$
* **Train-Val-Test Split:** $70-20-10$
* **Pre-Trained Weight:** imagenet
* **Fully Connected:** Pre-Trained Feature -> Flatten -> Dense ($256$, relu) -> Dropout ($0.5$) -> Dense ($512$, relu) -> Dense ($2$, softmax)
* **Optimizer:** Adaptive Momentum (Adam)
* **Learning Rate:** $0.0001$
* **Loss Function:** Categorical Crossentropy
* **Number of Epochs:** $20$

## Model Performance
Below are the results from the experiment:

| No  | Model       | Val Accuracy | Val Loss | Test Accuracy | Test Loss |
|-----|-------------|--------------|----------|---------------|-----------|
| 1   | InceptionV3 | $0.957$      | $0.190$  | $0.968$       | $0.155$   |
| 2   | Resnet50V2  | $0.932$      | $0.216$  | $0.934$       | $0.171$   |
| 3   | Xception    | $0.929$      | $0.202$  | $0.952$       | $0.194$   |

From the experiments conducted, based on the specified parameters, the transfer learning model with InceptionV3 as the pre-trained model yielded the best results compared to the others. Here are the accuracy, loss, and confusion matrix graphs from the InceptionV3 model:

![Loss Graph](https://github.com/emhaihsan/gender-classification/blob/main/img/loss.png)
![Accuracy Graph](https://github.com/emhaihsan/gender-classification/blob/main/img/akurasi.png)
![Confusion Matrix](https://github.com/emhaihsan/gender-classification/blob/main/img/confusionmatrix.png)
![Wrong Predictions](https://github.com/emhaihsan/gender-classification/blob/main/img/wrongprediction.png)

It can be seen that from the 20 epochs run, the model converged in the early epochs. This also happened with other models like Resnet and Xception. From this, at least two things can be assumed. First, it is possible that the learning rate of $10 \times 10^{-4}$ is already too high since the model uses pre-trained weights from imagenet. Second, it is possible that the model is already performing well and optimally, even though it does not seem to improve in terms of loss, the accuracy on test data has reached above 95%, so this assumption might be valid.

Furthermore, when looking at the results of incorrect predictions, the characteristics that are noticeable could also mislead humans, such as long hair, face shape, and the use of accessories. From this, it can be assumed that the errors made by the model might still fall into the reasonable category.

## Simple Application with Streamlit
To test if the model can be implemented into an application, a simple web app was created using [Streamlit](https://streamlit.io/). The appearance of the created application is as follows:

![Streamlit App](https://github.com/emhaihsan/gender-classification/blob/main/img/streamlit.png)

The code for the above implementation can be seen [here](https://github.com/emhaihsan/gender-classification/blob/main/genderclf.py).





