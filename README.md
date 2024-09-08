# Fact-and-Reflection (FaR) Improves Confidence Calibration of Large Language Models
This is the github repo for our ACL 2024 paper: Fact-and-Reflection (FaR) Improves Confidence Calibration of Large Language Models, [(Link)](https://aclanthology.org/2024.findings-acl.515.pdf).


The included Python file replicates the results in Section 3.4 of the paper: Generalizing to other language models, which shows how FaR shows good calibration over various open-source language models, with StrategyQA as the data and Token. Prob is the confidence extraction method.

Running the code mainly requires installing [vLLM](https://docs.vllm.ai/en/stable/getting_started/installation.html). Hugginface_hub is required for models with restricted usage. Do not forget to add the desired model to your token in the Hugging Face settings.

The authors are working on preparing an easy-to-use github repository. Do not hesitate to send an email if you need the raw code.


## Citation

      @inproceedings{zhao-etal-2024-fact,
          title = "Fact-and-Reflection ({F}a{R}) Improves Confidence Calibration of Large Language Models",
          author = "Zhao, Xinran  and
            Zhang, Hongming  and
            Pan, Xiaoman  and
            Yao, Wenlin  and
            Yu, Dong  and
            Wu, Tongshuang  and
            Chen, Jianshu",
          editor = "Ku, Lun-Wei  and
            Martins, Andre  and
            Srikumar, Vivek",
          booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
          month = aug,
          year = "2024",
          address = "Bangkok, Thailand and virtual meeting",
          publisher = "Association for Computational Linguistics",
          url = "https://aclanthology.org/2024.findings-acl.515",
          pages = "8702--8718",
      }

## Others
If you have any other questions about this repo, you are welcome to open an issue or send me an [email](mailto:xinranz3@andrew.cmu.edu), I will respond to that as soon as possible.
