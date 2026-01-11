import pandas as pd
import numpy as np
import ast
import joblib
import warnings
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter
from collections import defaultdict
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import argparse
def normalize(s: str) -> str:
    s = s.strip()
    # remove any leading/trailing quotes/backticks that may block the prefix match
    s = s.strip('\'"`')
    # remove a leading Hxx. prefix even if preceded by punctuation/spaces
    s = re.sub(r'^[\s\W]*H\d+\.\s*', '', s)
    # remove any backslashes or forward slashes inside the string
    s = re.sub(r'[\\/]', '', s)
    # remove trailing semicolons or commas left behind
    s = s.rstrip(';,').strip()
    return s

# %%
def extract_unique_issue_types(df, column="Combined Issue Types"):
    """
    Process a column of issue types where values are strings of labels
    separated by ';' or ','. Cleans, deduplicates, and normalizes labels.

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name containing issue type strings

    Returns:
        list[list[str]]: List of lists, each containing cleaned issue types
    """
    df = df.copy()
    df[column] = df[column].fillna("Other")

    all_issue_types = []
    not_english = "H10. The approach is tested only on [not English], so unclear if it will generalize to other languages"
    prefer = "The paper doesn't use [my preferred methodology], e.g., deep learning"
    for entry in df[column]:
        if not_english in entry:
            cleaned_text = re.sub(re.escape(not_english), "", entry).strip()

            # Step 2: Remove any leftover trailing semicolons or commas
            cleaned_text = re.sub(r"[;,]\s*$", "", cleaned_text).strip()

            # Step 3: Split by ; or , into a list
            phrases = [p.strip() for p in re.split(r"[;,]", cleaned_text) if p.strip()]
            phrases.append(not_english)
        elif prefer in entry:
            cleaned_text = re.sub(re.escape(prefer), "", entry).strip()

            # Step 2: Remove any leftover trailing semicolons or commas
            cleaned_text = re.sub(r"[;,]\s*$", "", cleaned_text).strip()

            # Step 3: Split by ; or , into a list
            phrases = [p.strip() for p in re.split(r"[;,]", cleaned_text) if p.strip()]
            phrases.append(prefer)
        else:
            phrases = re.split(r"[;,]", entry)

        # 2. Remove duplicates while preserving order
        unique_phrases = list(dict.fromkeys(phrase.strip() for phrase in phrases if phrase.strip()))

        # 3. Normalize: remove prefixes like "H1. " or "H23."
        normalized = [normalize(s) for s in unique_phrases]
        all_issue_types.append(normalized)
    df[column] = all_issue_types
    return df


# %%
def process_issue_segments(
    df, 
    issue_col="Combined Issue Types", 
    seg_col="Review Segment", 
    include_other=False
):
    """
    Process issue types and group segments by issue type.

    Args:
        df (pd.DataFrame): Input dataframe
        issue_col (str): Column with issue type strings (separated by ; or ,)
        seg_col (str): Column with text segments
        include_other (bool): If True, keep 'Other' as a valid issue type

    Returns:
        tuple:
            - parsed_issues (list[list[str]]): Row-wise list of issue types
            - issue_to_segments (dict): {issue_type: [segments]}
    """
    df = df.copy()
    df[issue_col] = df[issue_col].fillna("Other")

    parsed_issues = []
    issue_to_segments = {}
    not_english = "H10. The approach is tested only on [not English], so unclear if it will generalize to other languages"
    prefer = "The paper doesn't use [my preferred methodology], e.g., deep learning"
    for issue_str, segment in zip(df[issue_col], df[seg_col]):
        if not isinstance(segment, str) or len(segment.strip()) < 10:
            # Skip very short segments
            parsed_issues.append([])
            continue
        if not_english in issue_str:
            cleaned_text = re.sub(re.escape(not_english), "", issue_str).strip()

            # Step 2: Remove any leftover trailing semicolons or commas
            cleaned_text = re.sub(r"[;,]\s*$", "", cleaned_text).strip()

            # Step 3: Split by ; or , into a list
            phrases = [p.strip() for p in re.split(r"[;,]", cleaned_text) if p.strip()]
            phrases.append(not_english)
        elif prefer in issue_str:
            cleaned_text = re.sub(re.escape(prefer), "", issue_str).strip()

            # Step 2: Remove any leftover trailing semicolons or commas
            cleaned_text = re.sub(r"[;,]\s*$", "", cleaned_text).strip()

            # Step 3: Split by ; or , into a list
            phrases = [p.strip() for p in re.split(r"[;,]", cleaned_text) if p.strip()]
            phrases.append(prefer)
        
        else:
            phrases = re.split(r"[;,]", issue_str)

        # 2. Clean and normalize
        unique_phrases = list(dict.fromkeys(
            phrase.strip() for phrase in phrases if phrase.strip()
        ))
        normalized = [normalize(s) for s in unique_phrases]

        # 3. Handle inclusion/exclusion of "Other"
        if not include_other:
            normalized = [s for s in normalized if s != "Other"]

        parsed_issues.append(normalized)

        # 4. Add segment to dictionary for each issue
        for issue in normalized:
            issue_to_segments.setdefault(issue, []).append(segment)

    return parsed_issues, issue_to_segments


Abstract_prompt = """
You are given:
- An issue label: {issue}
- Why this issue is problematic: {problem}
- Review segments (from NLP paper reviews): {segments}
- Desired number of questions: 10

Your goal:
Extract abstract, high level features that determine whether each review segment reflects the specified issue. Express each feature as an objective Yes/No question.

Rules you MUST follow:
1) Use ONLY the provided segments. Do NOT rely on external knowledge or assumptions.
2) Work at the ABSTRACT level (features that could generalize across wording), not surface phrasing.
3) Every question must be answerable by inspecting a single segment in isolation.
4) Questions must be neutral (no leading language), atomic (one idea per question), and non-redundant.
5) Avoid double negatives, multi-part questions, subjective terms (“clearly”, “obviously”), or jargon unless it appears in the segments.
6) Prefer beginnings like “Does…”, “Is…”, “Are…”, “Has…”, “Do…”, “Can…”.
7) Keep each question concise (ideally ≤ 18 words).
8) Produce EXACTLY N questions. If N is not given, produce 10.
9) OUTPUT FORMAT: Return ONLY a Python list of strings, with double quotes, no code fences, no comments, no trailing commas.
10) Do NOT include any analysis, notes, or explanations in the output — only the list.

Step-by-step procedure:
A) Parse & inventory:
   - Scan the segments to note common patterns about claims, evidence, specificity, comparisons, citations, quantification, assumptions, scope/coverage, and consistency.
   - Identify cues that would indicate presence/absence of the issue {issue}.

B) Abstract feature mining:
   - Convert recurring patterns into abstract properties that can reflect the similarity between the review segments you are given
   - Ensure each property directly supports diagnosing {issue} as problematic because {problem}.

C) Draft discriminative Yes/No tests:
   - For each property, write a Yes/No question that a reviewer could answer from a single segment.
   - Ensure questions collectively cover: evidence/citations, specificity/locating, correctness/faithfulness, scope, quantification, reproducibility/actionability, internal consistency, and fairness/balance (as applicable).

D) Prune & refine:
   - Remove overlaps; keep the most general, segment-checkable forms.
   - Rewrite to be atomic, neutral, and concise.
   - Ensure each question distinguishes “issue present” vs “issue absent”.

E) Final checks (must pass all):
   - [ ] Exactly N questions.
   - [ ] All are Yes/No answerable from a single segment.
   - [ ] No duplicates or near-duplicates.
   - [ ] No restating {issue} verbatim; focus on testable properties.
   - [ ] Python list of strings ONLY, no extra text.

Now produce the final output.

"""




# In[2]:


classifier_name = "microsoft/phi-4"

tokenizer = AutoTokenizer.from_pretrained(classifier_name,trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained(classifier_name,trust_remote_code = True,device_map='auto')
def generate_response(prompt,llm = "microsoft/phi-4"):
    
    
#     weakness = row['Review Segment']
#     input_prompt = prompt.replace('{{weakness}}', weakness)
    input_prompt = prompt
    messages = [
    {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)

    # Decode response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response # remove prompt from response

import pickle
import re
def extract_features(issue_to_segments,feature_name,llm,prompt1=Abstract_prompt):
    problems = {
    "The results are not surprising": "Many findings seem obvious in retrospect, but this does not mean that the community is already aware of them and can use them as building blocks for future work. Some findings may seem intuitive but haven’t previously been tested empirically.",
    "The results contradict what I would expect": "You may be a victim of confirmation bias, and be unwilling to accept data contradicting your prior beliefs.",
    "The results are not novel": "If the paper claims e.g. a novel method, and you think you've seen this before - you need to provide a reference (note the policy on what counts as concurrent work). If you don't think that the paper is novel due to its contribution type (e.g. reproduction, reimplementation, analysis) — please note that they are in scope of the CFP and deserve a fair hearing.",
    "This has no precedent in the existing literature": "Believe it or not: papers that are more novel tend to be harder to publish. Reviewers may be unnecessarily conservative.",
    "The results do not surpass the latest SOTA": "SOTA results are neither necessary nor sufficient for a scientific contribution. An engineering paper could also offer improvements on other dimensions (efficiency, generalizability, interpretability, fairness, etc.) If the authors do not claim that their contribution achieves SOTA status, the lack thereof is not an issue.",
    "The results are negative": "The bias towards publishing only positive results is a known problem in many fields, and contributes to hype and overclaiming. If something systematically does not work where it could be expected to, the community does need to know about it.",
    "This method is too simple": "The goal is to solve the problem, not to solve it in a complex way. Simpler solutions are in fact preferable, as they are less brittle and easier to deploy in real-world settings.",
    "The paper doesn't use [my preferred methodology], e.g., deep learning": "NLP is an interdisciplinary field, relying on many kinds of contributions: models, resource, survey, data/linguistic/social analysis, position, and theory.",
    "The topic is too niche": "A main track paper may well make a big contribution to a narrow subfield.",
    "The approach is tested only on [not English], so unclear if it will generalize to other languages": "The same is true of NLP research that tests only on English. Monolingual work on any language is important both practically (methods and resources for that language) and theoretically (potentially contributing to a deeper understanding of language in general).",
    "The paper has language errors": "As long as the writing is clear enough, better scientific content should be more valuable than better journalistic skills.",
    "The paper is missing the [reference X]": "Per ACL policy, missing references to prior highly relevant work is a problem if such work was published (which is not the same as 'put on arXiv') 3+ months before the submission deadline. Otherwise, missing references belong in the 'suggestions' section, especially if they were only preprinted and not published. Note that for resubmissions, papers are only required to make comparisons to highly related relevant work published at least three months prior to the original submission deadline.",
    "The authors could also do [extra experiment X]": "It is always possible to come up with extra experiments and follow-up work. But a paper only needs to present sufficient evidence for the claim that the authors are making. Any other extra experiments are in the “nice-to-have” category and belong in the “suggestions” section rather than “reasons to reject.” This heuristic is particularly damaging for short papers. If you strongly believe that some specific extra comparison is required for the validity of the claim, you need to justify this in your review.",
    "The authors should compare to a 'closed' model X": "Requesting comparisons to closed-source models is only reasonable if it directly bears on the claim the authors are making. One can always say \"it would be interesting to see how ChatGPT does this\", but due to methodological problems such as test contamination and a general lack of information about 'closed' models, such comparisons may not be meaningful. Behind this kind of remark is often an implicit assumption that scientific questions can only be asked of the “best” models, but pursuing many important questions requires a greater degree of openness than is offered by many of today's “best” models.",
    "The authors should have done [X] instead": "A.k.a. 'I would have written this paper differently.' There are often several valid approaches to a given problem. This criticism applies only if the authors' choices prevent them from answering their research question, their framing is misleading, or the question is not worth asking. If not, then [X] is a comment or a suggestion, but not a 'weakness'.",
    "Limitations != weaknesses": "No paper is perfect, and most *CL venues now require a Limitations section. A good review should not just take the limitations and list them as weaknesses or reasons to reject. If the reviewer wishes to argue that acknowledged limitations completely invalidate the work, this should be appropriately motivated.",
    "Not enough Info": "",
    "X is not clear": "",
    "The formulation of X is wrong": "",
    "The contribution is not novel": "",
    "The paper is missing recent baselines": "",
    "X was done in the way Y": "",
    "The algorithm's interaction with dataset is problematic": "",
    "The paper is missing relevant references": ""
    }
    pattern = r"\[(.*?)\]"
    feature_dict = {}
    problematic_reason = "When a reviewer raises this issue about a paper without providing evidence or justification, this may indicate a lazy review."
    for issue_types in tqdm(issue_to_segments.keys(), desc="Extracting abstract features"):
        if issue_types == "":
            continue
        else:
            if problems[issue_types] != "":
                problematic_reason = problems[issue_types]
            else:
                problematic_reason = "When a reviewer raises this issue about a paper without providing evidence or justification on why, this may indicate a lazy review."
            review_segments = issue_to_segments[issue_types][:5]
            input_prompt = prompt1.format(issue=issue_types,problem=problematic_reason,segments=review_segments)
            response1 = generate_response(input_prompt,llm)
            
            match1 = re.search(pattern, response1, re.DOTALL)
            

            if match1:
                content = match1.group(1)
                # Extract individual strings inside quotes
                items = re.findall(r'"(.*?)"', content)
                print(items)
            
                feature_dict[issue_types] = items
    with open(feature_name, "wb") as f:
        pickle.dump(feature_dict, f)


    return feature_dict


def main():
    """
    Main function to extract abstract features from review segments using an LLM.
    
    Accepts command-line arguments:
        --data_file: Path to the review data file (TSV format)
        --llm_name: Name or path of the LLM model to use
    """
    parser = argparse.ArgumentParser(
        description="Extract abstract features from review segments using an LLM"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the review data file (TSV format expected with columns: Review Segment, Combined Issue Types)"
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        required=True,
        help="Name or path of the LLM model to use"
    )
    
    args = parser.parse_args()
    
    # Load the review data file
    print(f"Loading review data from: {args.data_file}")
    df = pd.read_csv(args.data_file, sep="\t")
    
    print(f"Using LLM: {args.llm_name}")
    
    # Extract and process issues
    print("Extracting unique issue types...")
    df = extract_unique_issue_types(df, column="Combined Issue Types")
    
    print("Processing issue segments...")
    parsed_issues, issue_to_segments = process_issue_segments(
        df,
        issue_col="Combined Issue Types",
        seg_col="Review Segment",
        include_other=False
    )
    
    # Generate output filename based on LLM name
    safe_llm_name = re.sub(r'[^a-zA-Z0-9_-]', '_', args.llm_name)
    output_file = f"abstract_features_{safe_llm_name}.pkl"
    
    print(f"Extracting abstract features...")
    feature_dict = extract_features(issue_to_segments, output_file, args.llm_name)
    print(f"Results saved to: {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
