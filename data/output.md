## Document 1

**Title:** Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models

**Authors:** Mosh Levy, Alon Jacoby, Yoav Goldberg

**Affiliations:** Bar-Ilan University, Allen Institute for AI

**Abstract:** This paper investigates how extending input lengths affects the reasoning capabilities of Large Language Models (LLMs). Despite advancements, the consistency of LLM performance across varying input lengths is not well understood. A novel QA reasoning framework is introduced to assess this impact, isolating input length effects through multiple versions of the same sample with varying padding. Findings indicate a significant degradation in reasoning performance at shorter input lengths than the models' maximum capacity. This trend is consistent across different dataset versions, with traditional next word prediction metrics negatively correlating with reasoning performance. The study identifies failure modes that could guide future research to address LLM limitations.

**Introduction:** Recent advancements in LLMs demonstrate impressive performance across various tasks, including complex reasoning. However, models often struggle with longer inputs, and previous studies have not adequately controlled for variables affecting performance. This research aims to clarify whether the degradation in performance is due to longer input requirements or task difficulty by systematically studying the impact of input length while controlling other factors.

**Methodology:** The study employs a novel QA reasoning framework to measure the effect of input length on model performance. Multiple versions of the same sample are created, each extended with padding of different lengths, types, and locations. This approach allows for a controlled examination of how input length influences reasoning capabilities.

**Key Findings:**
- LLMs exhibit notable performance degradation at shorter input lengths than their technical maximum.
- The degradation trend is observed across all dataset versions, albeit with varying intensities.
- Traditional metrics, such as next word prediction, show a negative correlation with reasoning performance on the dataset.
- Identified failure modes can inform future research directions to mitigate observed limitations in LLMs.

**Conclusion:** The study highlights the critical impact of input length on the reasoning performance of LLMs, suggesting that longer inputs do not necessarily enhance performance and may introduce challenges. The findings provide insights for future research aimed at improving LLM capabilities in reasoning tasks.

## Document 2

The research paper introduces the Flexible LENgth Question Answering dataset (FLenQA), designed to investigate how input length affects the reasoning capabilities of large language models (LLMs). The dataset consists of True/False questions based on two pieces of information, with variations in input length achieved by embedding these pieces within longer, irrelevant texts. The study aims to ensure that models must reason over the entire input to answer correctly, maintaining the relevance of the information while varying the context length.

Key findings include:

1. **Performance Degradation**: LLMs show a significant drop in accuracy (from 0.92 to 0.68) when the input length increases to 3000 tokens, indicating that longer inputs negatively impact reasoning capabilities.

2. **Embedding Locations**: The research explores how the placement of relevant information within the context affects model performance, revealing consistent trends of degradation across different experimental settings.

3. **Next-Word Prediction vs. Reasoning**: The study finds that next-word prediction performance does not correlate with reasoning performance on long inputs, suggesting that these tasks may engage different capabilities of the models.

4. **Chain-of-Thought (CoT) Prompting**: While CoT prompting improves performance on shorter inputs, it does not significantly mitigate the performance drop on longer inputs, with the exception of GPT-4, which shows an increasing gap between CoT and normal prompting as input length increases.

5. **Failure Modes**: The analysis identifies several failure modes in model responses, particularly a tendency to not follow the reasoning process required for tasks with longer inputs.

The dataset and findings contribute to understanding the limitations of LLMs in reasoning tasks as input length increases, highlighting the need for further research in this area.

## Document 3

The document discusses the development of the Flexible LENgth Question Answering dataset (FLenQA), which aims to evaluate reasoning capabilities in language models while controlling for input length. The key contributions and methodologies are summarized as follows:

### Core Contributions:
1. **Dataset Creation**: FLenQA is designed to test reasoning over texts that were not previously available, thereby preventing data contamination in evaluations.
2. **Task Design**: The dataset includes three reasoning tasks:
- **Monotone Relations (MonoRel)**: Involves comparing two person names on a monotone scale.
- **People In Rooms**: A new task introduced in this dataset.
- **Simplified Ruletaker**: A modified version of an existing task.

### Methodology:
- **Base Instances**: Each instance consists of:
1. An optional prefix (e.g., task introduction).
2. Two key paragraphs that are thematically coherent and start with key sentences necessary for solving the task.
3. An optional suffix (e.g., a question about the context).

- **Length Isolation**: The study imposes requirements to ensure that reasoning is independent of the sample length, with relevant spans remaining consistent across variations.

- **Natural Inputs**: Inputs are designed to reflect natural user prompts, ensuring cohesiveness at the paragraph level while varying input length.

- **Expansion of Sentences**: Simple sentences are expanded into coherent paragraphs using GPT-4, followed by manual verification to maintain naturality without introducing new information.

### Datasets and Tasks:
- Each task consists of 100 base instances, with variations in length, background texts, and fact dispersion.
- The label distribution for each task is balanced between "True" and "False" outcomes, ensuring that most base instances can be correctly solved by language models in their unexpanded forms.

### Key Results:
- The dataset and the code for generating it are released to facilitate future studies on reasoning capabilities in language models.

This structured approach aims to enhance the evaluation of reasoning in language models while addressing potential biases introduced by input length and data contamination.

## Document 4

The document discusses a dataset inspired by kinship relations and reasoning tasks, particularly focusing on two main tasks: People In Rooms (PIR) and a simplified version of the Ruletaker task.

### Key Contributions:
1. **New Relation Types**: The authors define a new set of relation types based on previous work by Sinha et al. (2018).
2. **PIR Task**: This task involves inferring whether a person is in a room with a specific property based on two key sentences. The dataset is generated programmatically using names from the Faker library and hand-crafted relations.
3. **Simplified Ruletaker Task**: This task involves logical reasoning where each instance consists of a logical rule, two facts, and a question. It aims to evaluate reasoning capabilities in language models.

### Methodology:
- **Data Generation**: The dataset is created by randomly selecting names and relations, ensuring that the properties of rooms are mutually exclusive to avoid ambiguity.
- **Input Length Variations**: The authors expand each base instance to various input lengths (250, 500, 1000, 2000, and 3000 tokens) by adding background text that is irrelevant to the question. This background text can be:
- **Duplicate**: Repeating the key paragraph to maintain the same information.
- **Similar**: Using paragraphs from other instances of the same task while avoiding contradictions.

### Datasets Used:
- The dataset is inspired by the bAbI set of tasks (Weston et al., 2016) and employs a similar structure for reasoning tasks.

### Key Results:
- The document notes that initial experiments indicate that most large language models (LLMs) struggle with instances involving multiple rules or more than two facts, highlighting the challenges in reasoning tasks.

This structured approach aims to enhance the understanding of reasoning in natural language processing by providing a clear framework for evaluating models on specific logical tasks.

## Document 5

The research paper discusses the construction of inputs for evaluating large language models (LLMs) in reasoning tasks, specifically focusing on the placement of key paragraphs within background text. The key contributions and findings are summarized as follows:

### Core Contributions:
1. **Input Construction**: The study introduces a method for integrating key paragraphs into background text sourced from the Books Corpus, ensuring that the key information is embedded within irrelevant padding text.
2. **Placement Strategies**: Four distinct strategies for the placement of key paragraphs are explored:
- **Key paragraphs first**: Key paragraphs are placed at the beginning.
- **Key paragraphs middle**: Key paragraphs are centered with padding on both sides.
- **Key paragraphs last**: Key paragraphs are positioned at the end.
- **Random placement**: Key paragraphs are interspersed randomly with padding.

3. **Baseline Accuracy Evaluation**: The paper evaluates the baseline accuracy of various LLMs using minimal text samples that include only the question and relevant key paragraphs. High accuracy (>0.89) was observed across most models, with GPT-3.5 being the lowest at 0.77.

### Methodology:
- The models evaluated include GPT-3.5, GPT-4, Gemini Pro, Mistral 70B, and Mixtral 8x7B.
- The performance was assessed using both direct prompts and chain-of-thought (CoT) prompting, with CoT generally improving accuracy.

### Key Results:
- The results indicate that all models performed well on minimal text inputs, with GPT-4 achieving perfect accuracy in several cases.
- The impact of input length on reasoning performance was also examined, revealing that even in scenarios with only relevant tokens, accuracy decreased with increased input length.

### Conclusion:
The findings suggest that while LLMs can achieve high accuracy with well-structured inputs, the length and placement of information significantly affect their reasoning capabilities. The study highlights the importance of input design in optimizing model performance for reasoning tasks.

## Document 6

The research paper investigates the impact of input length and the positioning of key paragraphs on the accuracy of language models (LLMs) in reasoning tasks. Key findings include:

1. **Accuracy Trends**: As the input length increases beyond 500 tokens, accuracy generally decreases. However, models like GPT-3.5 and GPT-4 show resilience to this degradation when the additional tokens are relevant.

2. **Positioning of Key Paragraphs**: The adjacency of key paragraphs significantly influences accuracy. Models tend to perform better when key paragraphs are positioned at the beginning or end of the text, with a noted recency bias when they appear last.

3. **Non-Adjacent Relevant Paragraphs**: The study also examines scenarios where relevant information is spread across non-adjacent sections. Results indicate a marked decline in performance as the length increases, particularly when models must gather evidence from distinct locations.

4. **Irrelevant Material**: The paper explores how the type of irrelevant text affects model performance. Surprisingly, models performed worse when irrelevant paragraphs were dissimilar to relevant ones, contrary to initial expectations that such differences would facilitate focus on relevant content.

5. **Padding Types**: The impact of different types of padding (irrelevant text) on accuracy was analyzed, revealing that models generally struggled more with irrelevant paragraphs that were different from the relevant ones.

Overall, the findings highlight the complexities of LLM performance in relation to input structure and content relevance, suggesting that both the length of input and the arrangement of key information are critical factors in model accuracy.

## Document 7

The section discusses the relationship between perplexity, next word prediction accuracy, and reasoning performance in large language models (LLMs), particularly in the context of Chain of Thought (CoT) prompting.

Key contributions and findings include:

1. **Perplexity and Downstream Tasks**: The paper highlights that model perplexity does not always correlate with performance on downstream tasks, as shown in previous studies (Liu et al., 2023a; Xia et al., 2022; Tay et al., 2022).

2. **Chain of Thought (CoT) Prompting**: CoT prompting encourages LLMs to articulate reasoning steps before arriving at an answer. This technique has been shown to improve accuracy in reasoning tasks (Kojima et al., 2022; Wei et al., 2022).

3. **Next Word Prediction Methodology**: The authors measure next word accuracy instead of perplexity due to limitations in closed models. They prompt models to predict the next word in a text and compare this to reasoning performance on the same samples.

4. **Results on CoT and Input Length**: The results indicate that while CoT improves performance for some models (e.g., GPT-4), it does not fully mitigate the decline in performance with longer inputs. In contrast, Gemini-Pro shows decreased performance with longer inputs despite improved accuracy on shorter ones.

5. **Failure Modes**: The authors identify four failure modes related to incorrect responses, including:
- **Failure to Answer**: Some models refuse to answer questions as input length increases.
- **Label Bias**: Certain models show a tendency to favor one label (often "false") as input length grows, despite a balanced dataset.

Overall, the findings suggest that next word prediction and perplexity are not reliable substitutes for evaluating reasoning performance in LLMs, especially with longer inputs. The full results and additional details are available in the appendix of the paper.

## Document 8

The research paper discusses the performance of large language models (LLMs) in generating answers using Chain-of-Thought (CoT) prompting, particularly focusing on how input length affects their ability to reason and provide accurate responses. Key findings include:

1. **Bias in Answer Generation**: The models exhibit a tendency to generate "False" answers more frequently than "True" ones, and they often produce non-answers that do not adhere to the prompt instructions.

2. **Impact of Input Length**: As the input length increases, the models' ability to locate relevant texts and perform reasoning degrades. This is evidenced by a decrease in the coverage of relevant facts in the CoT reasoning steps, leading to a higher incidence of incorrect responses.

3. **Early Responses**: The tendency for models to provide final answers before reasoning steps increases with longer inputs. This behavior contradicts the expected performance of autoregressive models, which should ideally attend to earlier tokens for reasoning.

4. **Statistical Dependence**: Incorrect responses are statistically linked to the occurrence of answers before reasoning steps, indicating a failure to follow prompt instructions as input length grows.

5. **Evaluation Methods**: The paper contrasts two evaluation pathways for LLMs: benchmarks of downstream tasks and next word prediction. While benchmark datasets provide fixed-length inputs, they limit understanding of how varying input lengths impact model performance.

Overall, the study highlights significant challenges in LLM performance related to input length and reasoning, suggesting that improvements in CoT prompting and model training may be necessary to enhance accuracy and adherence to instructions.

## Document 9

The paper investigates the impact of input length on the reasoning performance of Large Language Models (LLMs). It builds on previous research that explored various aspects of input intervention and task properties. The authors specifically focus on how extended input lengths affect model performance, revealing a significant drop in performance with longer inputs, even before reaching the models' maximum input capacity.

Key contributions include:

1. **Dataset Creation**: The authors constructed FLenQA, a dataset designed to isolate the length factor by adjusting irrelevant parts of the input, allowing for a focused analysis of input length effects.

2. **Findings**: The study demonstrates that regardless of sample adjustments, input length has a strong influence on reasoning performance. Specific failure modes were identified, such as challenges in following extended instructions and biases towards less relevant information.

3. **Evaluation Recommendations**: The authors argue for a more nuanced evaluation of LLMs, suggesting that performance should be assessed across various input lengths rather than a single length to better understand a model's capabilities.

4. **Future Directions**: The paper highlights unexplored aspects of LLM performance related to input length and suggests areas for future research to address identified weaknesses.

The study emphasizes the need for comprehensive evaluation methods to accurately assess LLM performance in reasoning tasks, particularly as input lengths vary.

## Document 10

The document appears to be a list of academic references related to language models and their capabilities, particularly in reasoning, summarization, and handling long contexts. Here are some key contributions and themes from the references:

1. **Human-like Reasoning in Language Models**: McClelland and Hill (2022) discuss how language models exhibit reasoning patterns similar to humans.

2. **Implicit Bias in Language Models**: Liu et al. (2023) explore how the pre-training loss affects downstream performance, emphasizing the importance of implicit bias.

3. **Long Context Handling**: Ding et al. (2024) introduce "Longrope," a method to extend the context window of language models beyond 2 million tokens, while Sainz et al. (2023) highlight the need to measure data contamination in NLP evaluations.

4. **Summarization Techniques**: Gidiotis and Tsoumakas (2020) propose a divide-and-conquer approach for summarizing long documents, and Liu et al. (2022) present an end-to-end segmentation-based method for news summarization.

5. **Zero-shot Reasoning**: Kojima et al. (2022) demonstrate that large language models can perform zero-shot reasoning tasks effectively.

6. **Impact of Context on Reasoning**: Jin et al. (2024) investigate how the length of reasoning steps affects the performance of large language models.

7. **Evaluation Challenges**: Several papers, including those by Jacovi et al. (2023) and Shaham et al. (2023), address the challenges of evaluating language models, particularly regarding data contamination and the need for standardized benchmarks.

8. **Distraction by Irrelevant Context**: Shi et al. (2023) find that large language models can be easily distracted by irrelevant information, impacting their performance.

These references collectively contribute to the understanding of language models' capabilities, limitations, and the methodologies for improving their performance in various tasks.

## Document 11

The document discusses various datasets and methodologies related to reasoning tasks in natural language processing, particularly focusing on the Ruletaker, MonoRel, and People in Rooms (PIR) tasks.

### Key Contributions:
1. **Ruletaker Task**: The study generates new, simpler samples for the Ruletaker task, which originally involved varying numbers of reasoning steps. The new samples consist of two facts and one logical rule, maintaining the essence of the original task while simplifying the complexity.

2. **MonoRel Task**: This task involves reasoning based on key paragraphs that describe a monotonic relationship between two individuals, with one individual being common to both descriptions. The paragraphs are embedded in padding text to create a mixture for the reasoning task.

3. **People in Rooms (PIR) Task**: In this task, one paragraph describes an individual's location, while another describes an attribute of that individual, facilitating reasoning about spatial and descriptive relationships.

### Methodology:
- Each task is constructed with 100 base instances, where each instance includes two paragraph-length texts (key paragraphs).
- The paragraphs are edited to ensure they are of similar lengths, with an average length of 125 tokens achieved by truncating sentences beyond a specific length.

### Datasets:
- The document provides statistics on the mean number of tokens for different padding types across various lengths (250, 500, 1000, 2000, 3000 tokens) for both the Ruletaker and MonoRel tasks.

### Key Results:
- The generated datasets are designed to ensure that reasoning over the input is necessary, thereby enhancing the evaluation of text-based reasoning capabilities in language models.

This work emphasizes the importance of inductive bias and the structure of model architectures in scaling language models for reasoning tasks.

## Document 12

The document discusses various prompt structures used in evaluating models for the People In Rooms (PIR) task. It outlines different types of prompts, including normal and chain-of-thought (CoT) formats, and specifies how to derive answers based on provided facts and rules. The evaluation setup includes parameters for reproducibility, such as temperature settings and configurations for model behavior. Additionally, it describes the methodology for locating answers in model responses, focusing on the last occurrence of "true" or "false" in the output.

Key contributions include the systematic approach to prompt design and the evaluation of model responses, which aim to enhance the accuracy and reliability of true/false question answering in the context of the PIR task.

## Document 13

The section discusses the evaluation of the coverage of key facts in outputs generated by a Chain of Thought (CoT) reasoning model. The evaluation method involved searching for case-insensitive matches of key sentences from input paragraphs within the model's output. Full coverage is defined as the presence of both key sentences in the CoT output. The reliability of this evaluation method was confirmed through a manual review of a random sample of 100 responses, which showed accuracy in all instances.

The results are presented in a graph (Figure 13) that illustrates the accuracy of the CoT outputs based on input length, comparing different models such as GPT-3.5, GPT-4, Mistral Medium, Mixtral 8x7B, and Gemini Pro. The graph indicates varying levels of accuracy across different input lengths and models, with specific attention to the performance of the CoT method in relation to the Ruletaker dataset.

## Document 14

The figures presented in the document illustrate the accuracy of various models based on input length for two different datasets: MonoRel and People In Rooms (PIR).

**Figure 14** shows the accuracy by input length for the MonoRel dataset, comparing models such as GPT-3.5, GPT-4, Mistral Medium, Mixtral 8x7B, and Gemini Pro. The accuracy is plotted against the number of tokens in the input, with different padding strategies (resampling padding and Books Corpus padding) indicated.

**Figure 15** presents similar information for the PIR dataset, again comparing the same models and padding strategies.

Key observations include:
- The accuracy tends to vary with input length, with notable differences between the models.
- The performance of models like GPT-4 and Gemini Pro is highlighted, suggesting they may perform better across different input lengths compared to others.

These figures provide insights into how model performance scales with input size and the effectiveness of different padding techniques.

## Document 15

The provided text appears to be a portion of a research paper discussing the performance of various language models, including GPT-3.5, GPT-4, Gemini Pro, Mixtral 8x7B, and Mistral Medium. The figures referenced (Figure 16 and Figure 17) likely illustrate the models' accuracy and biases in answer generation based on input length and the position of key paragraphs.

Key contributions and findings from this section include:

1. **Model Comparison**: The performance of different models is compared based on their accuracy in generating responses, with metrics likely indicating how well each model performs across varying input lengths.

2. **Input Length Impact**: The results suggest that the accuracy of the models may vary with the length of the input, as indicated by the x-axis labeled "Input length (# tokens)" in the figures.

3. **Biases in Responses**: Figure 17 highlights the frequency of responses categorized as True, False, or Other/Refused, indicating potential biases in how different models generate answers based on the context or position of the input data.

4. **Data Handling**: The mention of "irrelevant padding" suggests that the study examines how different types of padding (similar vs. dissimilar) affect model performance, which is crucial for understanding the robustness of these models in real-world applications.

Overall, the section emphasizes the importance of model architecture and input handling in determining the effectiveness of language models in generating accurate and unbiased responses.

