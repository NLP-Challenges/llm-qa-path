# LLM Question Answering

This repository hosts the code for the RAG-based question answering component of the Data Chatbot challenge. It focuses on using instruction-tuning to enhance various open-source language models for the specific task of question answering. The models developed are tested both quantitatively and qualitatively with a range of contexts and questions. The most effective model is subsequently integrated into the [Study Bot Implementation](https://github.com/NLP-Challenges/Study-Bot), serving as the foundation for its question-answering capabilities.

The fine-tuning process utilizes a Data Version Control (DVC) pipeline, ensuring reproducibility and streamlined workflow management.

Our models are trained on CUDA-enabled GPUs using the `bitsandbytes` library, which requires CUDA-compatible hardware for optimal performance. The fine-tuning configurations are managed through `params.yaml`, allowing for easy adjustments and experimentation.

The fine-tuned models are published under the [nlpchallenges organization on Hugging Face](https://huggingface.co/nlpchallenges). You can find the models specific to this project at [nlpchallenges/chatbot-qa-path](https://huggingface.co/nlpchallenges/chatbot-qa-path).

## Repository Structure

The structure of this repository is organized as follows:

```markdown
└── 📁llm-qa-path
    └── .env.template [ℹ️ Template for .env file]
    └── requirements.txt
    └── README.md
    └── 📁eda notebooks [ℹ️ Contains notebooks for exploratory data analysis]
    └── dvc.lock [ℹ️ DVC lock file to ensure reproducibility]
    └── dvc.yaml [ℹ️ DVC pipeline configuration file]
    └── params.yaml [ℹ️ DVC params file for run configs]
    └── 📁src
        └── 📁stages [ℹ️ DVC pipeline stages]
    └── 📁data [ℹ️ DVC data folder]
        └── 📁models [ℹ️ Contains the resulting models post-training]
        └── 📁processed [ℹ️ Contains DVC stages processed data]
        └── 📁raw [ℹ️ Contains input files for the DVC stages]
    └── evaluation_question_answering.ipynb [ℹ️ Notebook for evaluating the models (NPR MC2)]
    └── test_context.json
```

## Setup

### Prerequisites

1. **Python Environment**: Create a new virtual Python environment to ensure isolated package management.
2. **Installation**: Navigate to the repository's root directory and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Configuring Environment Variables

1. **Environment File**: Copy `.env.template` to a new file named `.env`.
2. **API Keys**: Add your API keys and tokens in the `.env` file:
   ```
   HF_ACCESS_TOKEN=your-token-here
   HF_ACCESS_TOKEN_WRITE=your-token-here
   OPENAI_API_KEY=your-key-here
   ```
   Replace `your-token-here` and `your-key-here` with your actual Hugging Face and OpenAI API keys.

### Using the DVC Pipeline

The repository uses a DVC pipeline for model fine-tuning. This approach allows for version control of data and models, ensuring reproducibility and efficient experimentation.

1. **Configuration**: The fine-tuning process is controlled by the `params.yaml` file. Adjust the parameters in this file to customize the training process.
2. **Running the Pipeline**: To run the DVC pipeline, use the following command:
   ```bash
   dvc repro
   ```
   This command will execute the pipeline stages defined in `dvc.yaml`, using the configurations specified in `params.yaml`.
   