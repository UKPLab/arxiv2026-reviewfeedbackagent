import pandas as pd
from vllm import LLM, SamplingParams
import argparse
import os
# Load data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
# Init vLLM model
  # or your model
sampling_params_sec = SamplingParams(temperature=0.5, top_p=0.9, max_tokens = 5)
sampling_params_bio = SamplingParams(temperature=0.5, top_p=0.9, max_tokens = 1)

def preprocess_labels(data):
    data = [x.replace('"','').lower()for x in data]
    #data = [x.replace('strengthes','strengths').replace('weaknesses', 'weakness') for x in data]
    return data

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



def create_section_classification_prompt(review, sentence: str) -> str:
        """
        Create prompt for section classification task
        """
        prompt = f"""You are a text classifier. Your task is to classify the given sentence into one of the following sections:

            1. summary_of_strengthes - Sentences that highlight positive aspects, advantages, or strong points about the review
            2. summary_of_weaknesses - Sentences that point out negative aspects, limitations, or areas for improvement about the review
            3. comments,_suggestions_and_typos - Typos or additional suggestions

        Review: {review}
        Sentence: {sentence}
        
        Answer can be one of the following options <summary_of_strengths/ summary_of_weaknesses/ comments_suggestions_and_typos>. Answer with **only** one of the options and nothing else.
        Section: """
        return prompt
    
def create_ner_tagging_prompt(review, sentence: str) -> str:
        """
        Create prompt for NER segment annotation task
        """
        prompt = f"""You are a Segment Tagger. Your task is to tag the sentence with one of these labels in context of the review. A segment consists of multiple sentences. The first sentence should be tagged with 'B' and the rest with 'I'. You should **carefully** look at where the sentence appears in the review and then perform the tagging:

            - B: Beginning of a segment.
            - I: Continuation of a segment. Always follows a 'B'.

        Review: {review}
        Sentence: {sentence}

        Answer can be one of the following options <B/ I>. Answer with **only** one of the options and nothing else.
        BIO tags: """
        return prompt
def format_prompt(tokenizer, all_prompts):
    all_messages = []
    for prompt in all_prompts:
        sample_dict = [{"role": "user", "content": prompt}]
        all_messages.append(sample_dict)
    
    #print(all_messages[0])
    formatted_prompts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in all_messages]
    #formatted_prompts = [f'{x}. The feedback for improvement is : ' for x in formatted_prompts]
    return formatted_prompts


def main(args):
    df = pd.read_csv(args.data_path, sep='\t')
    df = df[df['Review'].notna()]
    #df['Sentence'] = df['Sentence'].fillna('')
    #df=df.head(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype='half')
# Build prompts
    prompts_sec = []
    prompts_bio = []
    tracking_ids = []
    gold_sections = []
    gold_bios=[]

    for idx, row in df.iterrows():
        sentence = row['Sentence']
        section = row['Section']
        review = ast.literal_eval(row['Review'])
        #print(sentence)
        p_sec = create_section_classification_prompt(sentence, review[section])
        p_bio = create_ner_tagging_prompt(sentence, review[section])
        prompts_sec.append(p_sec)
        prompts_bio.append(p_bio)
        tracking_ids.append(idx)
        gold_sections.append(row['Desired_Section'])
        gold_bios.append(row['Segment_annotation'])
    # Run batch inference
    formatted_prompts_sec = format_prompt(tokenizer, prompts_sec)
    formatted_prompts_bio = format_prompt(tokenizer, prompts_bio)
    results_sec = llm.generate(formatted_prompts_sec, sampling_params_sec)
    results_bio = llm.generate(formatted_prompts_bio, sampling_params_bio)
# Parse results
    classified_sections = []
    bio_tags = []

    for res in results_sec:
        output = res.outputs[0].text.strip()
        #print(output)
        section = output
        #bio_line = next((l for l in output.splitlines() if l.lower().startswith("bio")), "")
    
        #section = section_line.split(":", 1)[-1].strip()
        #bio = bio_line.split(":", 1)[-1].strip()
    
    # Basic validation
        if section not in ['summary_of_strengthes', 'summary_of_weaknesses', 'comments,_suggestions_and_typos']:
            #print(section)
            section = 'summary_of_weaknesses'
            #section = section


        classified_sections.append(section)
    
    for res in results_bio:
        output = res.outputs[0].text.strip()
        #print(output)
        bio = output
        #bio_line = next((l for l in output.splitlines() if l.lower().startswith("bio")), "")
    
        #section = section_line.split(":", 1)[-1].strip()
        #bio = bio_line.split(":", 1)[-1].strip()
    
        if bio not in ['B', 'I', 'O']:
            #print(bio)
            bio = 'O'
            #bio = bio

        #classified_sections.append(section)
        bio_tags.append(bio)

# Add back to DataFrame
    df['classified_section'] = ''
    df['BIO'] = ''
    df.loc[tracking_ids, 'classified_section'] = classified_sections
    df.loc[tracking_ids, 'BIO'] = bio_tags
    df.to_csv(args.output_path, sep='\t', index=False)
    print(f"Saved classification with section and BIO tags to '{args.output_path}'")
    sec_acc, sec_f1, sec_prec, sec_rec, bio_acc, bio_f1, bio_prec, bio_rec = evaluation(gold_section=gold_sections, gold_bio=gold_bios, predicted_section=classified_sections, predicted_bio=bio_tags)
    results_path = os.path.splitext(args.output_path)[0] + '_results.txt'
    f = open(results_path, 'w')
    print(f'Section Accuracy: {sec_acc:.4f}, Section F1: {sec_f1:.4f}, Section Precision: {sec_prec:.4f}, Section Recall: {sec_rec:.4f}')
    print(f'BIO Accuracy: {bio_acc:.4f}, BIO F1: {bio_f1:.4f}, BIO Precision: {bio_prec:.4f}, BIO Recall: {bio_rec:.4f}')
    f.write(f'Section Accuracy: {sec_acc:.4f}, Section F1: {sec_f1:.4f}, Section Precision: {sec_prec:.4f}, Section Recall: {sec_rec:.4f}\n')
    f.write(f'BIO Accuracy: {bio_acc:.4f}, BIO F1: {bio_f1:.4f}, BIO Precision: {bio_prec:.4f}, BIO Recall: {bio_rec:.4f}\n')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model", type=str, required=True, help="Model name or path for vLLM")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output TSV file")
    args = parser.parse_args()
    main(args)