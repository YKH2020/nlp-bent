import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

def run_inference(model_dir: str, prompt: str) -> str:
    """
    Loads a fine-tuned model from disk and generates text based on the provided prompt.

    Args:
        model_dir (str): Directory where the fine-tuned model is saved.
        prompt (str): The input prompt for text generation.

    Returns:
        str: The generated text response.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=100,
        )
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return response

def main():

    old_model = "meta-llama/Llama-3.2-1B"
    # Run inference with the fine-tuned model
    inference_prompt = (
        "You are a high level Philosophy Researcher.\n"
        "Question: What is the journey a skeptic goes through when pondering new ideas?\n"
        "Task: Please answer this question concisely.\n\n"
        "Answer: "
    )
    print("\nInference with the original model:")
    response = run_inference(old_model, inference_prompt)

    response = response.replace(inference_prompt, " ")
    print("Response for the older model: ", response)

    new_model = "finetuned-philosophers-llama3.2-1b"

    print("\nInference with the fine-tuned model:")
    response = run_inference(new_model, inference_prompt)

    print("Response for the new model: ", response)
    

if __name__ == "__main__":
    main()
