import pandas as pd
from vllm import LLM, SamplingParams
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer
import ast
import os

# Sampling settings
sampling_params_sec = SamplingParams(temperature=0.5, top_p=0.9, max_tokens=5)
sampling_params_bio = SamplingParams(temperature=0.5, top_p=0.9, max_tokens=1)

def preprocess_labels(data):
    print([str(x).replace('"', '').lower() for x in data])
    return [str(x).replace('"', '').lower() for x in data]

def evaluation(gold_section, gold_bio, predicted_section, predicted_bio):
    sec_gold = preprocess_labels(gold_section)
    pred_gold = preprocess_labels(predicted_section)
    section_acc = accuracy_score(sec_gold, pred_gold)
    section_prec = precision_score(sec_gold, pred_gold, average='macro', zero_division=0)
    section_rec = recall_score(sec_gold, pred_gold, average='macro', zero_division=0)
    section_f1 = f1_score(sec_gold, pred_gold, average='micro')

    bio_acc = accuracy_score(gold_bio, predicted_bio)
    bio_prec = precision_score(gold_bio, predicted_bio, average='macro', zero_division=0)
    bio_rec = recall_score(gold_bio, predicted_bio, average='macro', zero_division=0)
    bio_f1 = f1_score(gold_bio, predicted_bio, average='micro')
    return section_acc, section_f1, section_prec, section_rec, bio_acc, bio_f1, bio_prec, bio_rec

def create_section_classification_prompt(review, sentence: str, prev_section_output: str = "") -> str:
    context = f"Previous sentence was classified as: {prev_section_output}" if prev_section_output else ""
    prompt = f"""You are a text classifier. Your task is to classify the given sentence into one of the following sections:

1. summary_of_strengthes - Sentences that highlight positive aspects, advantages, or strong points about the review
2. summary_of_weaknesses - Sentences that point out negative aspects, limitations, or areas for improvement about the review
3. comments,_suggestions_and_typos - Typos or additional suggestions

Review: {review}
{context}
Sentence: {sentence}

Answer can be one of the following options <summary_of_strengthes/ summary_of_weaknesses/ comments_suggestions_and_typos>. Answer with **only** one of the options and nothing else.
Section: """
    return prompt

def create_ner_tagging_prompt(review, sentence: str, prev_bio_output: str = "") -> str:
    context = f"Previous sentence was tagged as: {prev_bio_output}" if prev_bio_output else ""
    prompt = f"""You are a Segment Tagger. Your task is to tag the sentence with one of these labels in context of the review. A segment consists of multiple sentences. The first sentence should be tagged with 'B' and the rest with 'I'. You should **carefully** look at where the sentence appears in the review and then perform the tagging:

- B: Beginning of a segment.
- I: Continuation of a segment. Always follows a 'B'.

Review: {review}
{context}
Sentence: {sentence}

Answer can be one of the following options <B/ I/ O>. Answer with **only** one of the options and nothing else.
BIO tags: """
    return prompt

def format_prompt(tokenizer, all_prompts):
    all_messages = [[{"role": "user", "content": prompt}] for prompt in all_prompts]
    formatted_prompts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in all_messages]
    return formatted_prompts

def main(args):
    # Load and filter data
    df = pd.read_csv(args.data_path, sep='\t')
    df = df[df['Review'].notna() & df['Sentence'].notna()]
    df = df.reset_index(drop=True)
    #df = df.head(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype='half')

    # Outputs and tracking
    classified_sections = []
    bio_tags = []
    tracking_ids = []
    gold_sections = []
    gold_bios = []

    prev_section_output = ""
    prev_bio_output = ""

    for idx, row in df.iterrows():
        try:
            sentence = row['Sentence']
            section = row['Section']
            review = ast.literal_eval(row['Review'])

            if section not in review:
                print(f"Skipping row {idx} - section key '{section}' not in review.")
                continue

            prompt_sec = create_section_classification_prompt(review[section], sentence, prev_section_output)
            prompt_bio = create_ner_tagging_prompt(review[section], sentence, prev_bio_output)

            formatted_prompt_sec = format_prompt(tokenizer, [prompt_sec])[0]
            formatted_prompt_bio = format_prompt(tokenizer, [prompt_bio])[0]

            res_sec = llm.generate([formatted_prompt_sec], sampling_params_sec)[0]
            res_bio = llm.generate([formatted_prompt_bio], sampling_params_bio)[0]

            section_output = res_sec.outputs[0].text.strip()
            if section_output not in ['summary_of_strengthes', 'summary_of_weaknesses', 'comments,_suggestions_and_typos']:
                section_output = 'summary_of_weaknesses'

            bio_output = res_bio.outputs[0].text.strip()
            if bio_output not in ['B', 'I', 'O']:
                bio_output = 'O'

            classified_sections.append(section_output)
            bio_tags.append(bio_output)
            tracking_ids.append(idx)
            gold_sections.append(row['Desired_Section'])
            gold_bios.append(row['Segment_annotation'])

            prev_section_output = section_output
            prev_bio_output = bio_output

        except Exception as e:
            print(f"Skipping row {idx} due to error: {e}")
            continue

    # Final shape validation
    print(f"Gold Sections: {len(gold_sections)}, Predicted Sections: {len(classified_sections)}")
    print(f"Gold BIOs: {len(gold_bios)}, Predicted BIOs: {len(bio_tags)}")

    # Add predictions to dataframe
    df['classified_section'] = ''
    df['BIO'] = ''
    df.loc[tracking_ids, 'classified_section'] = classified_sections
    df.loc[tracking_ids, 'BIO'] = bio_tags

    # Save predictions
    df.to_csv(args.output_path, sep='\t', index=False)
    print(f"Saved classification results to: {args.output_path}")

    # Evaluation
    sec_acc, sec_f1, sec_prec, sec_rec, bio_acc, bio_f1, bio_prec, bio_rec = evaluation(gold_sections, gold_bios, classified_sections, bio_tags)

    results_path = os.path.splitext(args.output_path)[0] + '_results.txt'
    with open(results_path, 'w') as f:
        print(f'Section Accuracy: {sec_acc:.4f}, Section F1: {sec_f1:.4f}, Section Precision: {sec_prec:.4f}, Section Recall: {sec_rec:.4f}')
        print(f'BIO Accuracy: {bio_acc:.4f}, BIO F1: {bio_f1:.4f}, BIO Precision: {bio_prec:.4f}, BIO Recall: {bio_rec:.4f}')
        f.write(f'Section Accuracy: {sec_acc:.4f}, Section F1: {sec_f1:.4f}, Section Precision: {sec_prec:.4f}, Section Recall: {sec_rec:.4f}\n')
        f.write(f'BIO Accuracy: {bio_acc:.4f}, BIO F1: {bio_f1:.4f}, BIO Precision: {bio_prec:.4f}, BIO Recall: {bio_rec:.4f}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model", type=str, required=True, help="Model name or path for vLLM")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output TSV file")
    args = parser.parse_args()
    main(args)