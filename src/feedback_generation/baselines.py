import numpy as np
from vllm import LLM, SamplingParams
import textstat
import argparse

# -----------------------------
# Fitness Function (shared)
# -----------------------------
def compute_fitness(feedback, template):
    words = feedback.split()
    n_words = len(words)
    n_sent = feedback.count('.') + feedback.count('!') + feedback.count('?')

    sc_len = min(n_sent, 5) / 5
    template_tokens = set(template.split())
    fb_tokens = set(words)
    sc_temp = len(template_tokens & fb_tokens) / max(len(template_tokens), 1)
    sc_read = textstat.flesch_reading_ease(feedback)

    forbidden_terms = ["hi", "hello", "thanks", "good job", "nice work", "great paper"]
    pen_forb = sum(feedback.lower().count(t) for t in forbidden_terms) / max(n_words, 1)

    return sc_len + sc_temp + sc_read - pen_forb


# -----------------------------
# Baseline 1: 1-pass
# -----------------------------
def one_pass(llm, review_segment, issue):
    prompt = f"""
Generate constructive reviewer feedback for the following review segment.
Review Segment: {review_segment}
Identified Issue: {issue}

Feedback:
"""
    return llm.generate(
        [prompt],
        SamplingParams(temperature=0.7, max_output_tokens=200)
    )[0].text


# -----------------------------
# Baseline 2: Template-guided
# -----------------------------
def template_guided(llm, review_segment, issue, template):
    prompt = f"""
You are an expert at providing actionable feedback to improve the writing of review segments. Given a review
segment and its identified issue, your goal is to produce feedback that the reviewer can use to revise the segment.
You are also provided a feedback template for the identified issue.
Instructions: Adapt the provided feedback template to match the comment in the review segment. Ensure the feedback
is actionable, specific, and targeted to improving the segment.
Review Segment: {review_segment}
Identified Issue: {issue}
Template: {template}

Feedback:
"""
    return llm.generate(
        [prompt],
        SamplingParams(temperature=0.7, max_output_tokens=200)
    )[0].text


# -----------------------------
# Baseline 3: Plan-then-generate
# -----------------------------
def plan_then_generate(llm, review_segment, issue, template):
    plan_prompt = f"""
Create a short plan to address the issue in the review segment.
Review Segment: {review_segment}
Identified Issue: {issue}

Plan:
"""
    plan = llm.generate(
        [plan_prompt],
        SamplingParams(temperature=0.3, max_output_tokens=100)
    )[0].text

    gen_prompt = f"""
You are an expert at generating actionable
feedback to improve review segments.
Instructions: Given: 1. A review segment (weakness)
2. Identified issue(s) 3. A feedback template for the
issue(s) 4. A plan describing what to use as knowl-
edge sources (abstract, summary, template) to guide
feedback. 5. An explanation decsribing how to use
the knowledge sources.
Your task: - Generate actionable feedback for the
reviewer to improve the review segment. - Incorpo-
rate the plan into the template: adapt the template
to match the specific review segment and issue. -
Keep the feedback precise, relevant, and constructive.
- Output only the feedback text, do not include any
explanations or extra text.
Plan: {plan}
Template: {template}
Example Output: "Feedback: The reviewer should
clarify the novelty of the contribution by referencing
prior work; following the template, highlight which
claims are incremental and which are novel. [5]
Feedback:
"""
    return llm.generate(
        [gen_prompt],
        SamplingParams(temperature=0.7, max_output_tokens=200)
    )[0].text


# -----------------------------
# Baseline 4: Best-of-N (BoN)
# -----------------------------
def best_of_n(llm, review_segment, issue, template, n, n_gen):
    candidates = []
    for _ in range(n * n_gen):
        prompt = f"""
You are an expert at providing actionable feedback to improve the writing of review segments. Given a review
segment and its identified issue, your goal is to produce feedback that the reviewer can use to revise the segment.
You are also provided a feedback template for the identified issue.
Instructions: Adapt the provided feedback template to match the comment in the review segment. Ensure the feedback
is actionable, specific, and targeted to improving the segment.
Review Segment: {review_segment}
Identified Issue: {issue}
Template: {template}

Feedback:
"""
        fb = llm.generate(
            [prompt],
            SamplingParams(temperature=0.8, max_output_tokens=200)
        )[0].text
        candidates.append(fb)

    scores = [compute_fitness(fb, template) for fb in candidates]
    return candidates[np.argmax(scores)]


# -----------------------------
# Baseline 5: Self-Refinement
# -----------------------------
def self_refinement(llm, review_segment, issue, template, n, n_gen):
    refined = []
    for _ in range(n):
        prompt = f"""
System: You are an expert at generating and refining
actionable feedback for improving review segments.
Instructions: Given: 1. A review segment (weakness)
2. Identified issue(s) 3. An initial draft feedback
Your task: - Generate detailed feedback that helps
the reviewer improve the review segment. - Critically
evaluate the initial feedback and refine it to be more
precise, constructive, and actionable. - Ensure the
feedback is relevant to the identified issue(s) and
avoids vague statements. - Output only the refined
feedback text, do not include explanations, reasoning,
or any extra text.
Variables: - Review Segment: weakness - Identified
Issue: identified issue - Initial Feedback: initial feed-
back
Example Output: "Feedback: The reviewer should
provide concrete examples illustrating missing base-
lines; clearly indicate which comparisons are neces-
sary and why. [5]"
Review Segment: {review_segment}
Identified Issue: {issue}
Template: {template}

Feedback:
"""
        fb = llm.generate(
            [prompt],
            SamplingParams(temperature=0.7, max_output_tokens=200)
        )[0].text

        for _ in range(n_gen):
            refine_prompt = f"""
Refine the following feedback to be clearer, more concise, and more aligned with the template.
Template: {template}
Feedback: {fb}

Refined Feedback:
"""
            fb = llm.generate(
                [refine_prompt],
                SamplingParams(temperature=0.4, max_output_tokens=200)
            )[0].text

        refined.append(fb)

    scores = [compute_fitness(fb, template) for fb in refined]
    return refined[np.argmax(scores)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feedback generation baselines")
    parser.add_argument("--llm_model", type=str, default="gpt-4", help="LLM model name")
    parser.add_argument("--review_segment", type=str, required=True, help="Review segment text")
    parser.add_argument("--issue", type=str, required=True, help="Identified issue label")
    parser.add_argument("--template", type=str, required=True, help="Feedback template text")
    parser.add_argument("--baseline", type=str, default="all", choices=["1-pass","Temp","Plan","BoN","Self-Ref.","all"], help="Which baseline to run")
    parser.add_argument("--n", type=int, default=5, help="Number of candidates / population")
    parser.add_argument("--n_gen", type=int, default=3, help="Number of generations / refinement steps")

    args = parser.parse_args()

    llm = LLM(args.llm_model)

    # Run selected baseline(s)
    results = {}
    if args.baseline in ["1-pass","all"]:
        results["1-pass"] = one_pass(llm, args.review_segment, args.issue)
    if args.baseline in ["Temp","all"]:
        results["Temp"] = template_guided(llm, args.review_segment, args.issue, args.template)
    if args.baseline in ["Plan","all"]:
        results["Plan"] = plan_then_generate(llm, args.review_segment, args.issue, args.template)
    if args.baseline in ["BoN","all"]:
        results["BoN"] = best_of_n(llm, args.review_segment, args.issue, args.template, args.n, args.n_gen)
    if args.baseline in ["Self-Ref.","all"]:
        results["Self-Ref."] = self_refinement(llm, args.review_segment, args.issue, args.template, args.n, args.n_gen)

    # Print results
    for name, fb in results.items():
        print(f"\n=== {name} ===\n{fb}\n")