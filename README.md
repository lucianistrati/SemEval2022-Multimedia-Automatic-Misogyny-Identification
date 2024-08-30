# Multimedia Automatic Misogyny Identification (MAMI)

## Project Overview

This repository contains a collection of Python scripts and modules for various tasks, including data processing, model training, and analysis, particularly focused on machine learning, natural language processing, and explainability techniques. The code is structured within the `src` directory, which holds all the necessary files for running experiments, fine-tuning models, and analyzing results.

## Directory Structure

Below is an overview of the main scripts and modules present in the `src` directory:

### 1. **Data Processing and Feature Extraction**
   - **`analyze_data.py`**: Script for analyzing datasets, extracting insights, and preparing data for model training.
   - **`convert_to_binary.py`**: Converts datasets or labels into a binary format for specific model requirements.
   - **`dataset_loader.py`**: Loads datasets from various sources, including the Hugging Face dataset library.
   - **`feature_extractor.py`**: Extracts features from datasets, suitable for input into machine learning models.
   - **`meme_text_preproc.py`**: Pre-processes text data specifically for meme analysis tasks.
   - **`process_data.py`**: Handles general data processing tasks, including cleaning and normalization.
   - **`text_preprocess.py`**: Aids in the preprocessing of text data, such as tokenization and stopword removal.
   - **`vit_feat_extractor.py`**: Feature extraction using Vision Transformer (ViT) models for image data.
   - **`utils.py`**: Utility functions that support various data processing and analysis tasks.

### 2. **Model Training**
   - **`finetune.py`**: Fine-tunes pre-trained models on custom datasets.
   - **`train_adapter.py`**: Trains adapter layers on top of pre-trained models.
   - **`train_fasttext.py`**: Trains FastText models on text data.
   - **`train_linear_transformer.py`**: Implements training procedures for Linear Transformer models.
   - **`train_ml_model.py`**: General script for training traditional machine learning models.
   - **`train_perceiver.py`**: Training script for Perceiver models.
   - **`train_performer.py`**: Training script for Performer models.
   - **`train_reformer.py`**: Training script for Reformer models.
   - **`train_sinkhorn_transformer.py`**: Training script for Sinkhorn Transformer models.
   - **`train_transformer.py`**: Trains Transformer models for various tasks.
   - **`train_word2vec.py`**: Training script for Word2Vec models.
   - **`train_xlnet.py`**: Trains XLNet models.
   - **`train_xtransformer.py`**: Trains X-Transformer models.
   - **`vit_train.py`**: Training Vision Transformer (ViT) models.
   - **`xgb_model.py`**: Implements training and evaluation for XGBoost models.

### 3. **Analysis and Explainability**
   - **`analysis_alibi.py`**: Analyzes model outputs using Alibi, a library for explainability.
   - **`analysis_interpret_text.py`**: Text interpretation and explainability analysis.
   - **`analysis_lime.py`**: Analysis using LIME (Local Interpretable Model-Agnostic Explanations).
   - **`analysis_shap.py`**: SHAP (SHapley Additive exPlanations) based model analysis.
   - **`analysis_shifterator.py`**: Utilizes Shifterator for analyzing shifts in text data.
   - **`analysis_skater.py`**: Uses Skater for interpreting machine learning models.
   - **`analysis_xai.py`**: General explainable AI (XAI) analysis tools and techniques.
   - **`analysis_yellowbrick.py`**: Visualizes model performance using Yellowbrick.

### 4. **Specialized Scripts**
   - **`huggingface_datasets`**: Integration with Hugging Face datasets for easy data loading.
   - **`nn_utils.py`**: Neural network utility functions for model building and training.
   - **`ro_gpt.py`**: Implementation of a custom GPT model variant.
   - **`train_perceiver.py`**: Another script focused on training the Perceiver model.
   - **`vqa.py`**: Handles Visual Question Answering (VQA) tasks and model training.

## Getting Started

### Prerequisites

To run the scripts in this repository, you need the following installed on your system:

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SemEval2022-Multimedia-Automatic-Misogyny-Identification.git
   ```
2. Navigate to the `src` directory:
   ```bash
   cd SemEval2022-Multimedia-Automatic-Misogyny-Identification/src
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Each script is standalone and can be executed independently. For example, to train a Transformer model, you can run:

```bash
python train_transformer.py
```

Refer to the script files for specific usage details and arguments.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure your code follows the existing style and is well-documented.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Feel free to modify this README as per your project's specific needs!
