
input_text_relevance = '''<s> Task Description: 
You are given an instruction, a response to evaluate, and a score rubric representing an evaluation criterion.

Your goal:
1. Write detailed feedback that assesses the response strictly based on the rubric.
2. Assign a score between 1 and 5, referring to the rubric.
3. Output in the following format:
Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)
4. Do not include any opening, closing, or explanations.

Instruction to evaluate: You are an expert evaluating feedback generated for improving review segments. The feedback provided will help the reviewer improve the segment.

Weakness: {weakness}
Feedback to evaluate: {feedback}

Score Rubric:
Does the feedback directly relate to the content or structure of the review? Feedback should focus on aspects that help improve the review, including conciseness, reasoning, accuracy, or actionable corrections.

Score 1: Feedback is off-topic or unrelated to the review’s content or structure; provides no actionable guidance.
Score 2: Feedback mentions the review but mostly addresses minor or tangential points; limited actionable guidance.
Score 3: Feedback is partially relevant; identifies some issues but mixes relevant and loosely connected commentary; limited guidance.
Score 4: Feedback is clearly related to review content or structure; provides actionable suggestions addressing key issues.
Score 5: Feedback is tightly focused on review content or structure; comments are precise, actionable, and justified, improving clarity, reasoning, or accuracy.

Feedback:'''

input_text_constructiveness = '''<s> Task Description: 
You are given an instruction, a response to evaluate, and a score rubric representing an evaluation criterion.

Your goal:
1. Write detailed feedback that assesses the response strictly based on the rubric.
2. Assign a score between 1 and 5, referring to the rubric.
3. Output in the following format:
Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)
4. Do not include any opening, closing, or explanations.

Instruction to evaluate: You are an expert evaluating feedback generated for improving review segments. The feedback provided will help the reviewer improve the segment.

Weakness: {weakness}
Feedback to evaluate: {feedback}

Score Rubric:
Does the feedback offer suggestions for how the reviewer can improve? Constructive feedback should guide revisions, not just point out flaws, and can address both minor textual issues and substantive content.

Score 1: Feedback only identifies flaws without suggesting improvements; unhelpful or dismissive.
Score 2: Feedback includes vague or superficial advice; little actionable guidance.
Score 3: Feedback identifies issues and offers some guidance but partially unclear or limited.
Score 4: Feedback provides clear and relevant suggestions; mostly actionable, may lack some detail.
Score 5: Feedback consistently identifies issues and provides specific, practical, and helpful suggestions; supports targeted improvements effectively.

Feedback:'''


input_text_specificity = '''<s> Task Description: 
You are given an instruction, a response to evaluate, and a score rubric representing an evaluation criterion.

Your goal:
1. Write detailed feedback that assesses the response strictly based on the rubric.
2. Assign a score between 1 and 5, referring to the rubric.
3. Output in the following format:
Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)
4. Do not include any opening, closing, or explanations.

Instruction to evaluate: You are an expert evaluating feedback generated for improving review segments. The feedback provided will help the reviewer improve the segment.

Weakness: {weakness}
Feedback to evaluate: {feedback}

Score Rubric:
Does the feedback refer to specific parts or issues in the review? Feedback should clearly indicate what needs improvement, avoiding vague statements without context.

Score 1: Feedback is entirely vague or generic; no reference to particular parts of the review; not actionable.
Score 2: Feedback hints at an issue but lacks concrete references; reviewer cannot easily locate the problem.
Score 3: Feedback identifies general areas or sections but does not pinpoint exact sentences or claims; partially actionable.
Score 4: Feedback refers to specific sections or issues; minor vagueness may remain; mostly actionable.
Score 5: Feedback clearly identifies exact parts of the review and provides precise, actionable guidance, including minor textual or deeper content issues when justified.

Feedback:'''


input_text_conciseness = '''<s> Task Description: 
You are given an instruction, a response to evaluate, and a score rubric representing an evaluation criterion.

Your goal:
1. Write detailed feedback that assesses the response strictly based on the rubric.
2. Assign a score between 1 and 5, referring to the rubric.
3. Output in the following format:
Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)
4. Do not include any opening, closing, or explanations.

Instruction to evaluate: You are an expert evaluating feedback generated for improving review segments. The feedback provided will help the reviewer improve the segment.

Weakness: {weakness}
Feedback to evaluate: {feedback}

Score Rubric:
Does the feedback communicate suggestions concisely? Feedback should be brief, precise, and to the point, avoiding unnecessary verbosity while remaining actionable.

Score 1: Feedback is wordy, repetitive, or confusing; hard to extract actionable guidance.
Score 2: Feedback conveys ideas but is often verbose, vague, or partially actionable; contains unnecessary wording.
Score 3: Feedback is moderately concise, understandable, and partially actionable; some redundancy or extra wording may remain.
Score 4: Feedback is clear, focused, and mostly concise; communicates actionable guidance efficiently, with minor verbosity.
Score 5: Feedback is precise, concise, and easy to interpret; provides direct, actionable suggestions covering both minor and substantive issues effectively.

Feedback:'''