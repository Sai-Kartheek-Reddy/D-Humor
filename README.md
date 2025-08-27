# D-HUMOR: Dark Humor Understanding via Multimodal Open-ended Reasoning

**Accepted at IEEE ICDM 2025**  

---

# Abstract
Dark humor in online memes poses unique challenges due to its reliance on implicit, sensitive, and culturally contextual cues. To address the lack of resources and methods for detecting dark humor in multimodal content, we introduce a novel dataset of 4,397 Reddit memes annotated for dark humor, target category (gender, mental health, violence, race, disability, and other), and a three-level intensity rating (mild, moderate, severe). Building on this resource, we propose a reasoning-augmented framework that first generates structured explanations for each meme using a Large Vision–Language Model (VLM). Through a Role-Reversal Self-Loop, VLM adopts the author’s perspective to iteratively refine its explanations, ensuring completeness and alignment. We then extract textual features from both the OCR transcript and the self-refined reasoning via a text encoder, while visual features are obtained using a vision transformer. A Tri‐stream Cross‐Reasoning Network (TCRNet) fuses these three streams, text, image, and reasoning, via pairwise attention mechanisms, producing a unified representation for classification. Experimental results demonstrate that our approach outperforms strong baselines across three tasks: dark humor detection, target identification, and intensity prediction. The dataset, annotations, and code will be released to facilitate further research in multimodal humor understanding and content moderation.

## Core Contributions

- **Novel Dataset:** Proposed a new dataset on Dark Humor, collected from Reddit.  
- **Role-Reversal Self-Loop Refinement:** Introduced a prompting technique that makes LLMs think like the author of the post, improving LLM alignment for better explanation generation.  
- **Tri-stream Cross-Reasoning Network (TCRNet):** Developed a reasoning-augmented framework for enhanced understanding and processing of dark humor content.

## Subtasks in D-HUMOR Dataset

The D-HUMOR dataset includes three subtasks for evaluating dark humor understanding:

1. **Dark Humor Identification (Yes/No):** Binary classification of whether a post contains dark humor.  
2. **Target Identification:** 6-class classification of the target being addressed in the post:
   - Gender/Sex-Related Topics  
   - Mental Health  
   - Disability  
   - Race/Ethnicity  
   - Violence/Death  
   - Other (posts not falling under the above classes)  
3. **Intensity Classification:** Humor intensity levels: Mild (1), Moderate (2), Severe (3)  

## Overall Architecture of the Proposed Work

### Role-Reversal Self-Loop Prompting
We proposed a novel **Role-Reversal Self-Loop Prompting** technique for explanation generation via LLM alignment. The method uses an iterative self-loop where the LLM is prompted to think as the author of the post, enabling better understanding and alignment for generating explanations.  

![Role-Reversal Self-Loop](Images/Role-Reversal%20Self-Loop.png)

### Tri-stream Cross-Reasoning Network (TCRNet)
We also proposed a **reasoning-augmented framework**, **Tri-stream Cross-Reasoning Network (TCRNet)**, which fuses three streams: text, image, and reasoning via pairwise attention mechanisms. This produces a unified representation for classification.  

![TCRNet Architecture](Images/TCRNet%20Architecture.jpg)

Experimental results demonstrate that our approach outperforms strong baselines across three tasks.

## Dataset Access

Due to the sensitive nature of the content, access to the D-HUMOR dataset is **restricted to academic and research purposes only**.  

Please ensure compliance with the following terms before requesting access:  

- Use the dataset solely for non-commercial, research, or educational purposes.  
- Do not redistribute the dataset publicly.  

