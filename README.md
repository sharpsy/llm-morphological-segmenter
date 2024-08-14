# LLMSegm: Surface-level Morphological Segmentation Using Large Language Models

Welcome to the official code repository for the paper "LLMSegm: Surface-level Morphological Segmentation Using Large Language Models." 

## Overview
"LLMSegm" is a state-of-the-art surface-level morphological segmentation tool that leverages the capabilities of large language models (LLMs). This approach aids in understanding and analyzing the morphological structure of words in natural language processing (NLP) tasks. This repository contains the source code and instructions necessary to replicate the study results or to apply the segmentation tool to new datasets.

## Getting Started

### Prerequisites
- Docker (tested on rootless Docker)
- Make
- (alternatively, required Python packages are listed in `requirements.txt`)

### Setup
1. Clone the repository to your local machine using:
   ```
   git clone https://github.com/sharpsy/llm-morphological-segmenter.git
   ```
2. Navigate into the directory:
   ```
   cd llm-morphological-segmenter/
   ```
## Usage

You can use the following make command to build and run the segmentation model:

- **Run make - build docker container, train and evaluate the model**: 
  ```
  make all
  ```

## Citation
If you use any part of this code or the LLMSegm approach in your work, please cite our paper using the following BibTeX entry:

```bibtex
@inproceedings{pranjic-etal-2024-llmsegm,
    title = "{LLMS}egm: Surface-level Morphological Segmentation Using Large Language Model",
    author = "Pranji{\'c}, Marko  and
      Robnik-{\v{S}}ikonja, Marko  and
      Pollak, Senja",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.933",
    pages = "10665--10674"
}

```

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact
If you have any questions or feedback regarding the project, please raise an issue in this repository or contact the authors through the information provided in the paper.
