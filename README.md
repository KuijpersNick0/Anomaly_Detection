# Anomaly Detection on PCB Images

Welcome to the Anomaly Detection on PCB Images project! This project is part of my student job at a Belgian company where the goal is to analyze PCB (Printed Circuit Board) images produced by a pick-and-place machine. The focus is on detecting anomalies related to the components placed by the machine, including defects, incorrect orientation, upside-down placement, and insufficient adhesion.

## Objectives

- **Anomaly Detection:** Utilize state-of-the-art machine learning techniques to identify anomalies in PCB images, specifically focusing on component issues.

## Algorithm Overview

The anomaly detection algorithm is divided into two key sections:

### 1. Component Detection

In this stage, the algorithm extracts component placements from a CSV file provided to the machine. The `matchingTemplate` method of OpenCV (CV2) is employed to obtain a transformation matrix between the camera coordinates and CSV coordinates. By extracting the placements of components from this transformation, a robust dataset is prepared for further analysis.

### 2. CNN Classification

The extracted component placements are passed through a Convolutional Neural Network (CNN) model. This model classifies components into two categories: 'Good' or 'Bad.' This binary classification provides an initial assessment of the quality of component placements.

## Technologies Used

- TensorFlow
- PyTorch
- NumPy
- OpenCV (CV2)
- AWS SageMaker
- Pandas

## Contributors
Kuijpers Nick (@kuijpers2nick@gmail.com)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to contribute, report issues, or suggest improvements. Detect anomalies in PCB images effectively and contribute to ensuring high-quality PCB production !
