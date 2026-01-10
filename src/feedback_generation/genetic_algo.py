import argparse
import numpy as np
from vllm import LLM, SamplingParams
import textstat

# -----------------------------
# 1. Plan Generation
# -----------------------------
def generate_plan(llm, review_segment, issue, summary, strengths, template):
    """
    Generate a plan guiding feedback generation based on review segment, issue, and auxiliary info.
    """
    prompt = f"""
You are a planning agent. Given a review segment, the identified issue, and template guidance, 
write a plan describing how to integrate relevant info into the feedback.
Review Segment: {review_segment}
Identified Issue: {issue}
Reviewer Summary: {summary}
Noted Strengths: {strengths}
Template: {template}

Output a JSON-like plan with keys: 'knowledge_source' and 'justification'.
"""
    response = llm.generate([prompt], sampling_params=SamplingParams(temperature=0.3, max_output_tokens=200))
    return response[0].text

# -----------------------------
# 2. Population Initialization
# -----------------------------
def initialize_population(llm, review_segment, template, plan, n_candidates=10):
    """
    Generate initial population of feedback candidates based on template and plan.
    """
    population = []
    for _ in range(n_candidates):
        prompt = f"""
Generate professional, constructive feedback for the following review segment:
Review Segment: {review_segment}
Template: {template}
Plan: {plan}

Feedback:
"""
        response = llm.generate([prompt], sampling_params=SamplingParams(temperature=0.7, max_output_tokens=200))
        population.append(response[0].text)
    return population

# -----------------------------
# 3. Fitness Evaluation
# -----------------------------
def compute_fitness(feedback, template):
    """
    Compute intrinsic fitness of a feedback candidate.
    Conciseness, template adherence, readability, and forbidden term penalties.
    """
    words = feedback.split()
    n_words = len(words)
    n_sent = feedback.count('.') + feedback.count('!') + feedback.count('?')
    
    # Conciseness score
    n_max = 5  # max sentences considered ideal
    sc_len = min(n_sent, n_max) / n_max
    
    # Template adherence (simplified as n-gram overlap)
    template_ngrams = set(template.split())
    fb_ngrams = set(words)
    sc_temp = len(template_ngrams & fb_ngrams) / max(len(template_ngrams), 1)
    
    # Readability (Flesch reading score placeholder, normalized)
    sc_read = textstat.flesch_reading_ease(feedback)
    # Forbidden terms penalty
    forbidden_terms = ["hi", "hello", "thanks", "good job", "nice work", "great paper"]
    pen_forb = sum(feedback.lower().count(term) for term in forbidden_terms) / max(n_words, 1)
    
    fitness = sc_len + sc_temp + sc_read - pen_forb
    return fitness

# -----------------------------
# 4. Parent Selection & Crossover
# -----------------------------
def boltzmann_selection(candidates, fitnesses, n_parents=5, tau=1.0):
    """
    Select n_parents candidates based on fitness using Boltzmann (softmax) probabilities.
    """
    probs = np.exp(np.array(fitnesses)/tau)
    probs /= probs.sum()
    return list(np.random.choice(candidates, size=n_parents, p=probs, replace=False))

def crossover_multiple(llm, parents):
    """
    Generate a new feedback candidate by combining multiple parents.
    """
    parent_text = "\n".join([f"Parent {i+1}: {p}" for i, p in enumerate(parents)])
    prompt = f"""
Combine the following feedback candidates into a single, concise, professional, and constructive feedback:
{parent_text}
Ensure the resulting feedback integrates strengths from all parents and follows the template.
Feedback:
"""
    response = llm.generate([prompt], sampling_params=SamplingParams(temperature=0.7, max_output_tokens=200))
    return response[0].text

# -----------------------------
# 5. Evolutionary Loop
# -----------------------------
def evolutionary_feedback_generation(llm, review_segment, issue, summary, strengths, template,
                                     n_candidates=10, n_generations=3, n_parents=5):
    plan = generate_plan(llm, review_segment, issue, summary, strengths, template)
    population = initialize_population(llm, review_segment, template, plan, n_candidates)
    
    for _ in range(n_generations):
        fitnesses = [compute_fitness(fb, template) for fb in population]
        parents = boltzmann_selection(population, fitnesses, n_parents)
        child = crossover_multiple(llm, parents)
        population.append(child)
    
    fitnesses = [compute_fitness(fb, template) for fb in population]
    best_idx = np.argmax(fitnesses)
    return population[best_idx]

# -----------------------------
# 6. CLI Argument Support
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-driven Feedback Generation")
    parser.add_argument("--llm_model", type=str, default="gpt-4", help="Model to use for generation")
    parser.add_argument("--review_segment", type=str, required=True, help="Review segment to generate feedback for")
    parser.add_argument("--issue", type=str, required=True, help="Identified issue label")
    parser.add_argument("--summary", type=str, default="", help="Optional reviewer summary of the paper")
    parser.add_argument("--strengths", type=str, default="", help="Optional noted strengths")
    parser.add_argument("--template", type=str, required=True, help="Feedback template for the issue")
    parser.add_argument("--n_candidates", type=int, default=10, help="Number of initial candidates")
    parser.add_argument("--n_generations", type=int, default=3, help="Number of generations for evolution")
    parser.add_argument("--n_parents", type=int, default=5, help="Number of parents for crossover")
    
    args = parser.parse_args()
    
    llm = LLM(args.llm_model)
    
    best_feedback = evolutionary_feedback_generation(
        llm=llm,
        review_segment=args.review_segment,
        issue=args.issue,
        summary=args.summary,
        strengths=args.strengths,
        template=args.template,
        n_candidates=args.n_candidates,
        n_generations=args.n_generations,
        n_parents=args.n_parents
    )
    
    print("Best Feedback:\n", best_feedback)
