import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
from prompt_prometheus import (
    input_text_conciseness,
    input_text_constructiveness,
    input_text_relevance,
    input_text_specificity
)

def replace_input(row, input_text, args):
    weakness = 'Review Segment'
    outputs = 'outputs'
    if ';' in row[weakness]:
        row[weakness]=row[weakness].split(';')[0]
    return input_text.replace('{weakness}', row[weakness]).replace('{feedback}', row[outputs])

def prepare_batches(df, prompt_template, batch_size, args):
    inputs = [replace_input(row, prompt_template, args) for _, row in df.iterrows()]
    return [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]

def batch_generate(tokenizer, model, input_texts, max_new_tokens, device):
    batch_messages = [
        [
            {"role": "system", "content": "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance"},
            {"role": "user", "content": prompt}
        ]
        for prompt in input_texts
    ]

    messages_text = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=False,  # Don't tokenize yet
        add_generation_prompt=True
    )

    encoded = tokenizer(
        messages_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    

    outputs = model.generate(
        input_ids=encoded['input_ids'],
        attention_mask=encoded['attention_mask'],
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.03,
        pad_token_id=tokenizer.pad_token_id
    )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    feedbacks, scores = [], []

    for output in decoded_outputs:
        try:
            split_part = output.split('###')[-1]
            feedback, score = split_part.split('[RESULT]')
        except ValueError:
            feedback, score = "", ""
        feedbacks.append(feedback.strip().replace('\n', ' '))
        scores.append(score.strip().replace('\n', ' '))

    return feedbacks, scores

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    model.eval()

    df = pd.read_csv(args.input_path, sep='\t').dropna()
    #df = df.head(1)
    batch_size = args.batch_size or 4

    all_feedbacks = { "Conciseness": [], "Relevance": [], "Constructiveness": [], "Specificity": [] }
    all_scores = { "Conciseness": [], "Relevance": [], "Constructiveness": [], "Specificity": [] }

    dimensions = [
        ("Conciseness", input_text_conciseness),
        ("Relevance", input_text_relevance),
        ("Constructiveness", input_text_constructiveness),
        ("Specificity", input_text_specificity)
    ]

    for name, template in dimensions:
        print(f"Processing: {name}")
        batches = prepare_batches(df, template, batch_size, args)
        for batch in batches:
            feedbacks, scores = batch_generate(tokenizer, model, batch, args.max_new_tokens, device)
            all_feedbacks[name].extend(feedbacks)
            all_scores[name].extend(scores)

    df['Conciseness Feedback'] = all_feedbacks['Conciseness']
    df['Conciseness Score'] = all_scores['Conciseness']
    df['Relevance Feedback'] = all_feedbacks['Relevance']
    df['Relevance Score'] = all_scores['Relevance']
    df['Constructiveness Feedback'] = all_feedbacks['Constructiveness']
    df['Constructiveness Score'] = all_scores['Constructiveness']
    df['Specificity Feedback'] = all_feedbacks['Specificity']
    df['Specificity Score'] = all_scores['Specificity']
    df['Prompt'] = [args.prompt_type]*len(df)

    df.to_csv(f'{args.output_path}_{args.prompt_type}.csv', sep='\t', index=False)
    print(f"\n✅ Saved output to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--prompt_type', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(args)
