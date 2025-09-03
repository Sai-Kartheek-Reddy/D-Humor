# Role Reversal Self-Loop Prompting (RR-SLP)

We introduce Role Reversal Self-Loop Prompting (RR-SLP), a novel technique to improve LLM alignment for meme explanation tasks.
Instead of treating model outputs as final, RR-SLP re-injects the model’s own explanation back into the prompt but asks it to assume the role of the meme’s original author.
From this perspective, the model critiques and corrects its prior explanation, leading to finer-grained narrative understanding, better humor interpretation, and context-sensitive corrections.

This iterative role-reversal loop was critical in achieving higher explanation quality and interpretability, especially for subjective tasks like dark humor and meme analysis.

---

### 🔍 How RR-SLP Works
1. **Iteration 1**: Generate the explanation **normally** (baseline prompt).  
2. **Iterations ≥ 2**: Use **RR-SLP** to refine the explanation by having the LLM critique and rewrite the previous explanation from the meme author’s point of view.  

### 🔧 Inputs
- **Meme** 
- **Previous Explanation**: Generated in the previous iteration, with fields like Summary, Implied Joke, Narrative Structure, Emotional Effect, Dark Attributes, and Target.  

### **Role Reversal Self-Loop Prompt Template**  
>
> **You are the author and creator of the given dark humor meme.** 
> **Someone else has provided a detailed explanation of your meme’s meaning, humor, and components, including the Meme Summary, Implied Joke, Narrative Structure, Emotional Effect, Dark Attributes, and Target.**
>  
> **Your task is to review their explanation from your perspective as the original author. Analyze how well their reasoning aligns with your intended humor, message, tone, and overall context. For each component, identify any inaccuracies, missing details, or misunderstandings. Then, provide a thorough, corrected explanation that fully reflects your original intent, ensuring clarity, accuracy, and completeness.**  
>  
> **Please present your revised explanation below.**
>
> **[Explanation from Previous Iteration Inserted Here]**

# TCRNet - Multimodal Meme Classification

This folder contains the **implementation of TCRNet**, a multimodal model for meme classification using **images**, **meme text**, and **explanations**.  
The script is designed for easy experimentation across:  
1. **Dark Humor Prediction** (Binary)  
2. **Target Identification** (6-class)  
3. **Intensity Prediction** (3-class)  

---

## 📂 File
| File               | Description                                  |
|--------------------|----------------------------------------------|
| `TCRNet.py`        | Full training and evaluation script for TCRNet.|

---

## 🧩 Model Overview
TCRNet (**Tri-Stream Cross-Reasoning Network**) uses **three parallel encoders** and **cross-attention fusion**:

- **Inputs**:  
  - Meme text (BERT encoder)  
  - Meme image (ViT encoder)  
  - Explanations (SentenceTransformer encoder)  
- **Fusion**: Cross-attention layers align features from all three streams.  
- **Classifier**: Fully connected layers with task-specific output size.


```text
       Meme Text ──► BERT Encoder ──► Projection ──┐
                                                   │
       Meme Image ─► ViT Encoder ──► Projection ───┼──► Cross-Attention Fusion ─► Classifier ─► Prediction
                                                   │
  Explanation ──► SentenceTransformer ─► Projection┘
```
## Task Setup
Set the `TASK` and adjust labels in the dataset:

| Task         | Label Column          | Classes |
|--------------|----------------------|---------|
| Dark Humor   | `Dark`               | 2       |
| Target       | `Target`             | 6       |
| Intensity    | `Intensity`          | 3       |

---

## How to Run

### 1. Install dependencies
```bash
cd Code
pip install -r requirements.txt
```
### 2. Run the Script
```bash
python3 TCRNet.py
```
---
