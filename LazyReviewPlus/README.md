# LazyReviewPlus Dataset

This folder contains the **LazyReviewPlus** dataset and the template knowledge base used for the feedback generation experiments in the paper **"Reviewing the Reviewer: Elevating Peer Review Quality through LLM-Guided Feedback"**.

## Files Description

### 1. `template_knowledge.json`

This JSON file serves as the knowledge base for the feedback generation module. It maps each identified "lazy thinking" issue to a specific feedback template and the required knowledge source (e.g., Abstract, Paper Summary) needed to address it.

**Fields:**
- **Key**: The specific issue identifier and description (e.g., `"H1. The results are not surprising"`).
- **Value**: A dictionary containing:
  - **`Knowledge`**: Specifies what information source the LLM should utilize (e.g., `"Paper summary"`, `"Abstract"`, `"Template"`).
  - **`Template`**: A string template providing the structure and tone for the constructive feedback. It includes placeholders like `[insert reviewer comment here]`.

**Example:**
```json
"H7. This method is too simple": {
    "Knowledge": "Abstract",
    "Template": "Your comment — \"[insert reviewer comment here]\" — suggests the method is too simple. However, the goal is to solve the problem effectively..."
}
```

### 2. Dataset File 

The dataset file contains the annotated review sentences that can be used for training and evaluation. It includes labels for lazy thinking issues, segmentation tags, and review metadata.


#### ARR2022.tsv
**Key Columns:**

- **`ID`**: A unique identifier for the data point.
- **`Review`**: The full context of the review, often stored as a string representation of a dictionary, allowing the model to understand the segment's placement.
- **`Section`**: The key indicating which part of the review the sentence comes from. (e.g., `summary_of_weaknesses`, `summary_of_strengths`, `comments,_suggestions_and_typos`).
- **`Sentence`**: The text content of the review segment or sentence being analyzed.
- **`Desired_Section`**: The section of the review where the segment should belong to (e.g., `summary_of_weaknesses`, `summary_of_strengths`, `comments,_suggestions_and_typos`).
- **`Segment_annotation`**: BIO (Beginning, Inside, Outside) tags used for the segmentation task to identify argumentative units.
- **`Issue Type`**: The ground truth labels for lazy thinking / specificity issues associated with the segment. Multiple issues are separated by delimiters (e.g., `;` or `,`).
- **`full_reviews`**: The complete text of the reviews associated with the paper or submission.

#### EMNLP_final.tsv

**Key Columns:**
- **`ID`**: A unique identifier for the data point.
- **`Review`**: The full context of the review, often stored as a string representation of a dictionary, allowing the model to understand the segment's placement.
- **`Section`**: The key indicating which part of the review the sentence comes from. (e.g., `summary_of_weaknesses`, `summary_of_strengths`, `comments,_suggestions_and_typos`).
- **`Sentence`**: The text content of the review segment or sentence being analyzed.
- **`Desired_Section`**: The section of the review where the segment should belong to (e.g., `summary_of_weaknesses`, `summary_of_strengths`, `comments,_suggestions_and_typos`).
- **`Segment_annotation`**: BIO (Beginning, Inside, Outside) tags used for the segmentation task to identify argumentative units.
- **`Issue Type`**: The ground truth labels for lazy thinking / specificity issues associated with the segment. Multiple issues are separated by delimiters (e.g., `;` or `,`).
- **`full_reviews`**: The complete text of the reviews associated with the paper or submission.
