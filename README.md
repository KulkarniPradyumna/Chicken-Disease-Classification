# Chicken-Disease-Classification--Project


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml

In this project, I developed a robust and scalable chicken disease classification system using Convolutional Neural Networks (CNNs). The goal was to accurately identify various diseases in chickens based on images, enabling early detection and prompt intervention for better farm management.

Key Accomplishments:

<strong>CNN Algorithm Implementation: </strong>Leveraging the power of deep learning, I designed and implemented a CNN architecture to perform image classification. The model was trained on a diverse dataset of chicken disease images, achieving an impressive accuracy of [mention accuracy percentage].

<strong>Modular Coding and YAML Files:</strong> To ensure maintainability and scalability, I adopted a modular coding approach. The project was organized into reusable and independent modules, enhancing code readability and simplifying future enhancements. Additionally, I utilized YAML files for streamlined configuration management, making it easy to update model parameters and hyperparameters.

<strong>DVC for Data Version Control:</strong> Managing large-scale datasets is crucial in machine learning projects. To maintain a clean and version-controlled dataset, I integrated DVC into the project workflow. DVC allowed us to efficiently track changes to the dataset, collaborate with team members, and ensure reproducibility in data preparation.

<strong>Pipeline Development:</strong> I established an end-to-end data pipeline to facilitate seamless data flow from data preprocessing to model training and evaluation. The pipeline efficiently handled data augmentation, normalization, and batch processing, contributing to improved model generalization.

<strong>Continuous Integration and Continuous Deployment (CI/CD):</strong> By implementing CI/CD pipelines, I automated the testing and deployment processes, significantly reducing development cycle time. Continuous integration ensured smooth code integration and validation, while continuous deployment allowed for seamless model updates and enhancements.

<strong>Dockerization:</strong> To enable consistent deployment across different environments, I containerized the entire application using Docker. This ensured that the model and all dependencies were encapsulated, guaranteeing reproducibility and portability.

<strong>Deployment and API Development:</strong> The final model was deployed on a cloud-based platform, making it easily accessible for real-time inference. I developed a user-friendly API to allow external systems to interact with the model, enabling poultry farmers and veterinarians to diagnose chicken diseases conveniently.

<strong>Technologies Used:</strong>

Python
TensorFlow and Keras
YAML configuration
DVC (Data Version Control)
Docker
Cloud Services (e.g., AWS, Azure, Google Cloud)
API Development (e.g., Flask)

The chicken disease classification project exemplifies my proficiency in leveraging cutting-edge technologies, modular coding practices, efficient data version control, and deployment strategies to create practical solutions in the field of data-driven agriculture. The project's success in achieving high accuracy and seamless deployment highlights my expertise in developing end-to-end machine learning applications.


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/entbappy/Chicken-Disease-Classification--Project
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnncls python=3.8 -y
```

```bash
conda activate cnncls
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag




