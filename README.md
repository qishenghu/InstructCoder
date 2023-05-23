# CodeInstruct

![Alt text](./fig/demo.png "Pipeline & Example")

This repo hosts the dataset, code for the CodeInstruct project. The project presents the CodeInstruct dataset designed to adapt language models (LLMs) for code editing. 

Blog: https://blog.nus.edu.sg/kaixinli/2023/05/23/codeinstruct/


# Overview
Code editing encompasses a variety of pragmatic tasks that developers deal with daily. Despite its relevance and practical usefulness, automatic code editing remains an underexplored area in the evolution of deep learning models, partly due to data scarcity. To this end, we introduce CodeInstruct, a dataset designed to adapt LLMs for code editing, containing high-diversity code-editing tasks. It consists of over 114,000 instruction-input-output triplets and covers multiple distinct code editing scenarios. 



# Data Collection
To generate instructional data for code editing, we employed a similar method to Self-Instruct. This methodology of generating training data using LLMs requires minimal human-labeled data as seed tasks while still maintaining the quality and relevance of the tasks in the dataset. CodeInstruct is systematically expanded through an iterative process that commences with editing data sourced from GitHub commits as seed tasks. Seed and generated tasks are used subsequently bootstrapped to prompt ChatGPT for more task data. 




# Release
We are planning to release the following assets:
- [x] Full dataset: Over 114,000 code-editing instructional data.
- [x] Train / Validation data: Arround 95% / 5% of the full dataset.
- [x] Github seed / test data: A total of 768 github commit data.
- [x] Additional seed data: 592 generated samples in early stage.
- [ ] TODO: Source Code
- [ ] TODO: Checkpoints