# Agent for Philosophical Ideologies

## Problem Statement
As we technologically advance, we often stray from the humanities and focus more on practical fields. However, there should still be emphasis on core aspects of the humanities, like philosophy, as we often unknowingly use prevalent themes in the field to drive our moral and ethical code in advancement today. Our goal is to construct a model to better comprehend philsophical thoughts, and be able to interact with the prompter to provide insights that may be hard to come by online or difficult to digest from dense articles.

## Data
Our data was utilized from the following source: https://huggingface.co/datasets/AiresPucrs/stanford-encyclopedia-philosophy. Stanford Encyclopedia data on philosophy was chunked and labeled with philsophical related headers. This data was then cropped and used to refine our model responses in its related philosophical areas.

## Models

### Naive Approach
For Naive Approach, we choose the original language model Llama3.2-1B totally from https://huggingface.co/meta-llama/Llama-3.2-1B. It is a lightweight variant in the Llama series of language models, built with 1 billion parameters. With the auto-regressive language model that uses an optimized transformer architecture, it can delivery quality language understanding and generation while keeping resources usage low. It is an ideal choice for applications on hardware with limited computational power (e.g. CPU). 

### Traditional Approach

### Deep Learning Approach
Despite the lightweight and its efficiency in text generation of LLama3.2-1B. The comparably small parameters make it a bottleneck for having enough domain knowledge about philosophy ideas. Besides, the abstract and Intricately related
concepts of philosophy make reasoning of those ideas an important part for models to truly understand the philosophical ideas, which might not be founded in LLama3.2-1B due to its small size. To make the model build solider knowledge repository of philosophy and enhance the reasoning of different philosophical ideas, we fine-tuned the Llama3.2-1B model on the Stanford Encyclopedia data to feed philosophical ideas with corresponding schools of it. We used a structured prompt to relate the philosophical ideas with their schools. We guided the model to identify the corresponding school of each idea using the optimized prompt and do the fine-tuning with in-context learning approach to enhance the text-generation ability. 

## Evaluation Strategy
We decided to take several approaches for evaluation; primarily human evaluation as well as automated evaluation.

### Human Evaluation
We had one of our team members act as a subject matter expert for philosophical concepts. This allowed for the model to answer philosophy questions and explain its reasoning so that the human could determine whether the explanation is valid or not.

### Automated Evaluation
The automated evaluation strategy involved building a ground truth dataset that covers a variety of questions to assess multiple areas of the model. The exam was divided into multiple sections such as open ended questions, multiple choice questions, fixing incorrect statements, as well as adversarial questions. Each one of these methods allowed for the model to be assessed in multiple ways.

Although, this did prove to be challenging as it matters deeply how questions are structured as they need to be self-contained in order for the model to have the best chance at providing a clear answer. This involves building a dataset with deep knowledge of the field which we are not, therefore we were only able to make a small dataset to get some idea of how the model would perform.

For open-ended questions which measured explainability, we created a reference answer that is expected of the model which contains key concepts that the model should answer with. Then based on the model's output and the reference response, we calculated the BLEU and ROUGE scores to get an idea of the similarity between the reference and actual output of the model.

MCQ measured correctness of the model as it forces the model to make a choice and be evaluated concretely. This was also challenging as the options had to be carefully crafted to ensure that the model actually understands concepts rather than providing the most obvious answer.

Adversarial questions and Logical consistency questions were not created in the interest of time and lack of domain expertise. However, the goal was to assess robustness and contradictions respectively to further understand if the model understand philosophical concepts.

Overall, this provided a general structure to how the evaluation of this LLM could be done and compare the fine tuned model with the base model to evaluate if the fine tuning improved performance or not. All scores on this exam were quite low, likely indicating a lack of quality data, or a sparse model, or a combination of both.

## How to Run
How to run:
1) Clone the Repo:
```git clone https://github.com/YKH2020/nlp-bent.git```
2) Set up the environment:
```pip install -r requirements.txt```
3) Run training process of LLM:
```python llm-train.py```
After running the training process, the model weights will be saved in a folder named finetuned-philosophers-llama3.2-1b directly in this project path.
4) Run inference process of LLM:
You can download the folder with the model weights from google drive:
https://drive.google.com/drive/folders/10YoDmCBdTZnnRFJLNuguBjsCcojf7jLk?usp=drive_link
and save it under the project folder with the same name of this folder.
5) If you finish the training process, the model weights will be saved to this project folder directly. 
6) Run this command for inference process:
```python llm-inference.py```
You can design the prompt of the naive llama2.3-1B model and the fine-tuned model and compare the response of each model. 
