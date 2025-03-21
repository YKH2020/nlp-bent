from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd
from inference import run_inference
from huggingface_hub import login
from dotenv import load_dotenv
import os
    
def test_open_ended():
    '''
    This function tests the open ended questions with BLEU and ROUGE scores
    '''
    questions = pd.read_csv("evaluation_datasets/open_ended_evaluation.csv")

    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    inference_prompt = (
            "You are a high level Philosophy Researcher.\n"
            "Task: Please answer the following question.\n\n"
        )

    for index, row in questions.iterrows():
        
        question = row["Prompt"]
        prompt = inference_prompt +  "Question: " + question + "\n Answer: "
        
        reference_text = row["Reference response"]

        generated_text = run_inference("meta-llama/Llama-3.2-1B",  prompt)

        generated_text = generated_text.replace(prompt,"")

        reference_tokens = reference_text.split()
        generated_tokens = generated_text.split()

        smooth = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference_tokens, generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu_scores.append(bleu_score)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_score = scorer.score(reference_text, generated_text)
        rouge1_scores.append(rouge_score)
        rouge2_scores.append(rouge_score)
        rougeL_scores.append(rouge_score)


    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    rouge1_precisions = []
    rouge1_recall = []
    rouge1_fmeasure = []

    rouge2_precisions = []
    rouge2_recall = []
    rouge2_fmeasure = []

    rougeL_precisions = []
    rougeL_recall = []
    rougeL_fmeasure = []

    for i in range(len(rouge1_scores)):
        rouge1_precisions.append(rouge1_scores[i]["rouge1"].precision)    
        rouge1_recall.append(rouge1_scores[i]["rouge1"].recall)
        rouge1_fmeasure.append(rouge1_scores[i]["rouge1"].fmeasure)

        rouge2_precisions.append(rouge2_scores[i]["rouge2"].precision)
        rouge2_recall.append(rouge2_scores[i]["rouge2"].recall)
        rouge2_fmeasure.append(rouge2_scores[i]["rouge2"].fmeasure)

        rougeL_precisions.append(rougeL_scores[i]["rougeL"].precision)
        rougeL_recall.append(rougeL_scores[i]["rougeL"].recall)
        rougeL_fmeasure.append(rougeL_scores[i]["rougeL"].fmeasure)


    avg_rouge1_precision = sum(rouge1_precisions) / len(rouge1_precisions)
    avg_rouge2_precision = sum(rouge2_precisions) / len(rouge2_precisions)
    avg_rougeL_precision = sum(rougeL_precisions) / len(rougeL_precisions)

    avg_rouge1_recall= sum(rouge1_recall) / len(rouge1_recall)
    avg_rouge2_recall = sum(rouge2_recall) / len(rouge2_recall)
    avg_rougeL_recall = sum(rougeL_recall) / len(rougeL_recall)

    avg_rouge1_fmeasure = sum(rouge1_fmeasure) / len(rouge1_fmeasure)
    avg_rouge2_fmeasure = sum(rouge2_fmeasure) / len(rouge2_fmeasure)
    avg_rougeL_fmeasure = sum(rougeL_fmeasure) / len(rougeL_fmeasure)


    print("Avg BLEU Score:", avg_bleu_score)
    print("Avg Rouge1 Precision:", avg_rouge1_precision)
    print("Avg Rouge2 Precision:", avg_rouge2_precision)
    print("Avg RougeL Precision:", avg_rougeL_precision)

    print("Avg Rouge1 Recall:", avg_rouge1_recall)
    print("Avg Rouge2 Recall:", avg_rouge2_recall)
    print("Avg RougeL Recall:", avg_rougeL_recall)

    print("Avg Rouge1 F1:", avg_rouge1_fmeasure)
    print("Avg Rouge2 F1:", avg_rouge2_fmeasure)
    print("Avg RougeL F1:", avg_rougeL_fmeasure)

def test_mcq():
    """
    This function evaluates the MCQ abilities of the fine tuned model
    """
    mcq = pd.read_csv("evaluation_datasets/mcq_evaluation.csv")

    inference_prompt = (
        "You are a high level Philosophy Researcher.\n"
        "Task: Please answer the following question with ONLY the multiple choice letter in your answer.\n\n"
        "IMPORTANT: You MUST ANSWER WITH A or B or C or D\n"
    )

    correct = 0
    for index, row in mcq.iterrows():
        question = row["Question"]
        answer = row["Answer"]

        prompt = inference_prompt + "Question: " + question+"\nAnswer:  "
        generated_text = run_inference("model weights", prompt)

        generated_text = generated_text.replace(prompt,"")
        
        if generated_text.strip().lower() == answer.strip().lower():
            correct += 1

    score = correct / len(mcq)

    print("MCQ Score: ", score)

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_KEY")
    token=api_key
    login(token)

    test_open_ended()
    test_mcq()
