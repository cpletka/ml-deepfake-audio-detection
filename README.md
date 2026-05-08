# Deepfake Audio Detection

## Objective
This project evaluates multiple machine learning and deep learning models for detecting synthetic (deepfake) audio using benchmark and real-world datasets.

## Approach

### Data Processing
- Audio Normalization
- Noise handling/cleaning
- Dataset split

### Feature Extraction
- MFCC
- Spectrograms
- Temporal Features

### Models Tested
- Logistic Regression
- Random Forest
- Resnet
- 1D CNN
- CNN
- CRNN

### Evaluation Metrics
- Accuracy / Precision
- Recall / F1 Score

## Results

### ASVSpoof 2019 LA
| Model | F1-Score | Accuracy |
|---|---|---|
| Logistic Regression | 96% | 87% |
| Random Forest | 94% | 90% |
| Resnet | 63% | 56% |
| 1D CNN | 94% | 91% |
| CNN Spectrogram | 90% | 89% |
| CRNN | 84% | 76% |

### In The Wild
| Model | F1-Score | Accuracy |
|---|---|---|
| CRNN | 97% | 98% |
| 1D CNN | 96% | 97% |

## Practical Use Cases
- Banking
- Customer Service

## Team
- Eric Rodgers and Claudia Pletka
