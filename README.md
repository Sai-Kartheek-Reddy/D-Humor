# D-HUMOR: Dark Humor Understanding via Multimodal Open-ended Reasoning

**Accepted at IEEE ICDM 2025**  
*Status: Yet to be published*

---

# Abstract
Dark humor in online memes poses unique challenges due to its reliance on implicit, sensitive, and culturally contextual cues. To address the lack of resources and methods for detecting dark humor in multimodal content, we introduce a novel dataset of 4,397 Reddit memes annotated for dark humor, target category (gender, mental health, violence, race, disability, and other), and a three-level intensity rating (mild, moderate, severe). Building on this resource, we propose a reasoning-augmented framework that first generates structured explanations for each meme using a Large Vision–Language Model (VLM). Through a Role-Reversal Self-Loop, VLM adopts the author’s perspective to iteratively refine its explanations, ensuring completeness and alignment. We then extract textual features from both the OCR transcript and the self-refined reasoning via a text encoder, while visual features are obtained using a vision transformer. A Tri‐stream Cross‐Reasoning Network (TCRNet) fuses these three streams, text, image, and reasoning, via pairwise attention mechanisms, producing a unified representation for classification. Experimental results demonstrate that our approach outperforms strong baselines across three tasks: dark humor detection, target identification, and intensity prediction. The dataset, annotations, and code will be released to facilitate further research in multimodal humor understanding and content moderation.
