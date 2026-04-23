
# bLLeQA: Benchmarking LLMs for Grounded Legal Question-Answering in French and Dutch

This repository contains the code for reproducing the experimental results presented in the paper ["bLLeQA: Benchmarking LLMs for Grounded Legal Question-Answering in French and Dutch"]()


Retrieval-augmented generation (RAG) systems can play an important role in making law more accessible. However, large and reliable resources for training and benchmarking such systems remain scarce, especially for under-resourced languages like Dutch. To address this gap, and building on previous work (Louis et al., 2024), we introduce bLLeQA, a bilingual parallel question-answering dataset grounded in Belgian legal resources, both in French and Dutch. The dataset contains aligned questions, answers, and supporting articles in both languages, enabling evaluation of both retrieval and end-to-end RAG pipelines. Using bLLeQA, we benchmark the full RAG pipeline in a zero-shot setting, covering retrieval, citation extraction, refusal behavior, and generation quality. Our experiments show that open-weight models are competitive with proprietary models in retrieval and citation extraction, but lag behind in generation quality in the RAG pipeline. Across all models, refusal capability remains weak, meaning that models do not reliably detect when the provided supporting sources are incomplete. In addition, the end-to-end RAG setup still yields a substantial share of flawed responses, reaching 20% even in the best-case scenario.

## Documentation

Detailed documentation on the dataset and how to reproduce the main experimental results can be found [here](docs/README.md).

## Citation

For attribution in academic contexts, please cite this work as:

```latex
@inproceedings{
    banar2026blleqa,
    title={b{LL}e{QA}: Benchmarking {LLM}s for Grounded Legal Question-Answering in French and Dutch},
    author={Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Marija Kliocaite and Walter Daelemans},
    booktitle={4th Workshop on Towards Knowledgeable Foundation Models at ACL 2026},
    year={2026},
    url={https://openreview.net/forum?id=WBON1oFQ6d}
    }
```

## License

This repository is MIT-licensed.