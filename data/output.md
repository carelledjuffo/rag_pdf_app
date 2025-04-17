## Document 1

**Title:** Long Context RAG Performance of Large Language Models

**Authors:** Quinn Leng, Jacob Portes, Sam Havens, Matei Zaharia, Michael Carbin (Databricks Mosaic Research)

**Abstract:** This paper investigates the performance of Retrieval Augmented Generation (RAG) in the context of Large Language Models (LLMs) with extended context lengths. The study examines 20 popular LLMs, varying context lengths from 2,000 to 128,000 tokens, and up to 2 million tokens when feasible, across three domain-specific datasets. Key findings indicate that while retrieving more documents can enhance performance, only a few state-of-the-art LLMs maintain consistent accuracy beyond 64k tokens. The research also identifies specific failure modes in long context scenarios, highlighting areas for future exploration.

**Introduction:** The emergence of LLMs with longer context lengths, such as Anthropic Claude (200k tokens) and GPT-4-turbo (128k tokens), raises questions about their potential to replace traditional RAG workflows. This study aims to empirically assess how increased context length affects RAG performance and to identify the challenges associated with long context applications. RAG enhances LLM accuracy by integrating external information, which is beneficial across various applications.

**Key Contributions:**
- Comprehensive evaluation of RAG performance across 20 LLMs with varying context lengths.
- Insights into the benefits and limitations of long context in RAG applications.
- Identification of failure modes in long context scenarios, suggesting directions for future research.

**Methodology:** The study involved running RAG workflows on selected LLMs while systematically varying the context length. The performance was assessed using three domain-specific datasets.

**Results:** The findings reveal that while increased document retrieval can improve performance, only a limited number of advanced LLMs can sustain accuracy at context lengths exceeding 64k tokens. The research highlights distinct failure modes that occur in these long context scenarios.

**Conclusion:** The study underscores the potential of long context LLMs in RAG applications while also pointing out significant limitations and areas for further investigation.

## Document 2

The paper investigates the performance of various large language models (LLMs) in long-context retrieval-augmented generation (RAG) tasks. It evaluates 20 popular models, including both open-source and commercial options, across context lengths ranging from 2,000 to 2 million tokens. The study aims to determine how effectively these models can be utilized in RAG systems, particularly as context lengths increase.

Key findings include:

1. **Performance Variation with Context Length**: The study reveals that longer context does not consistently enhance RAG performance. Most models initially improve in performance with increased context length but then experience a decline. Only a few state-of-the-art models maintain accuracy beyond 64,000 tokens.

2. **Unique Failure Modes**: Different models exhibit distinct failure modes at long context lengths. Some provide incorrect answers, while others may fail to follow instructions or decline to answer due to concerns about copyright.

The paper also discusses the background of RAG, highlighting its dual-phase process of retrieval and generation, and notes the advancements in LLMs that allow for larger context lengths, with examples of models capable of handling up to 2 million tokens.

Overall, the research contributes to understanding the limitations and capabilities of LLMs in long-context RAG applications, emphasizing the need for careful consideration of context length in model selection and deployment.

## Document 3

The research paper discusses the performance of various large language models (LLMs) in the context of retrieval-augmented generation (RAG) tasks, particularly focusing on how the length of context affects their performance. Here are the core contributions, methods, datasets, and key results:

### Core Contributions
1. **Evaluation of Long Context Models**: The study evaluates the performance of 20 popular open-source and commercial LLMs with varying context lengths, addressing the limitations identified in previous studies regarding long context models.
2. **Analysis of RAG Performance**: It provides insights into how the number of retrieved document chunks influences the generation performance of these models, revealing that longer contexts do not uniformly enhance performance.

### Methodology
- **Models Evaluated**: The study includes models such as o1-mini, o1-preview, Gemini 1.5 Pro, GPT-4o, Claude 3.5, and several others, representing a mix of commercial and open-source options.
- **Datasets Used**: The experiments were conducted on three datasets: Databricks DocsQA, FinanceBench, and Natural Questions.
- **Retrieval Process**: Document chunks were retrieved using OpenAI's text-embedding-3-large model, with a chunk size of 512 tokens and a stride of 256 tokens, stored in a FAISS vector store.
- **Context Length Variation**: The context length was varied from 2,000 tokens to 128,000 tokens, and up to 2 million tokens when possible, to assess the impact on performance.
- **Evaluation Method**: The generation performance was judged by a calibrated "LLM-as-a-judge" using GPT-4o, and failure patterns were analyzed for selected models.

### Key Results
- **Performance Trends**: The best commercial models (e.g., o1-mini, GPT-4o, Claude 3.5 Sonnet) showed steady performance improvement with increasing context length. In contrast, many open-source models exhibited a performance increase followed by a decrease as context length increased.
- **Effective Context Length**: The findings suggest that the effective context length for many models is shorter than the maximum context length they claim to support, corroborating previous studies on the limitations of long context models.

This research highlights the nuanced relationship between context length and model performance, emphasizing the need for careful evaluation of LLM capabilities in practical applications.

## Document 4

The research paper discusses the performance of various language models (LLMs) on the FinanceBench dataset, particularly focusing on their ability to handle long context retrieval-augmented generation (RAG) tasks. Key findings include:

1. **Model Performance**: The OpenAI o1 models demonstrated significant improvements over previous versions like GPT-4 and GPT-4o, maintaining high answer correctness even at extreme context lengths (up to 128,000 tokens). In contrast, Google’s Gemini 1.5 Pro and Flash models, while having lower overall accuracy, showed consistent performance at very long contexts (up to 2,000,000 tokens).

2. **Context Length Limitations**: Most open-source models, such as Llama 3.1 and Qwen 2, exhibited a decline in performance beyond 16k-32k tokens, indicating limitations in handling long contexts effectively.

3. **Failure Patterns**: The study identified distinct failure modes among models when dealing with long contexts. For instance, Claude 3 Sonnet often refused to answer due to copyright concerns, while Gemini 1.5 Pro faced issues with overly sensitive safety filters. Other models like DBRX struggled with instruction adherence at longer contexts.

4. **Conclusion**: The findings suggest that while some models can leverage long contexts to enhance RAG performance, many still face significant challenges, particularly in maintaining accuracy and following instructions as context length increases.

Overall, the paper highlights the evolving capabilities of LLMs in handling long contexts and the need for further improvements in model design to address identified failure modes.

## Document 5

The document presents a failure analysis of various language models (LMs) on the Natural Questions (NQ) dataset, specifically focusing on models such as Gemini 1.5 Pro, Claude 3 Sonnet, Mixtral 8x7B, and Llama 3.1 405B. Key findings include:

1. **Failure Categories**: The analysis categorizes failures into types such as "wrong answer," "refusal," "fail_follow_inst," "task_failed," and "random_content." Notably, Gemini 1.5 Pro struggled with long context lengths due to overly sensitive safety filters, while Claude 3 Sonnet often refused to answer due to perceived copyright concerns.

2. **Context Length Impact**: The performance of LMs generally improved with longer context lengths up to 16-32k tokens, but this was not uniform across all models. The document suggests that models trained primarily on short contexts may not perform well with longer contexts, indicating a potential misalignment in training objectives.

3. **Retrieval-Augmented Generation (RAG)**: The results imply that for datasets smaller than 128k tokens, it might be feasible to bypass the retrieval step in a RAG pipeline by directly feeding the entire dataset into the LLM. However, this approach could be costly and may not yield optimal performance.

4. **Cost Analysis**: The document provides a cost comparison for processing queries with maximum sequence lengths of 128k tokens across different models, highlighting significant variations in costs. For instance, GPT-4o costs $0.32 per query, while Gemini 1.5 Pro costs $0.16. The analysis notes that using long contexts for RAG is more expensive than maintaining a vector database for document retrieval.

5. **Future Considerations**: The authors suggest that while the costs of using long contexts are currently high, advancements in batch inference and corpus caching may help mitigate these expenses. They also note a significant decrease in the cost per million input tokens for models like GPT-4 over the past year, indicating a trend towards more affordable long-context processing in the future.

Overall, the findings emphasize the complexities of model performance with varying context lengths and the economic implications of using long contexts in language model applications.

## Document 6

The acknowledgments section of the research paper expresses gratitude to individuals and teams who contributed to the experiments and provided feedback. Key contributors mentioned include Andrew Drozdov, Andy Zhang, and Erica Yuen, along with the Databricks AI Research team for their support and discussions. The research was funded by Databricks, and all experiments were conducted on the Databricks Mosaic AI platform.

Additionally, the document references earlier versions of the work that were published as blog posts, highlighting their titles and publication dates.

The references section lists various academic papers and resources relevant to the research, including technical reports and preprints from notable authors and institutions in the field of machine learning and AI. These references cover topics such as model capabilities, retrieval-enhanced machine learning, and semantic parsing, indicating a broad engagement with current research trends and methodologies.

## Document 7

The document appears to be a list of references from an academic paper, focusing on various studies related to language models, retrieval-augmented generation, and long-context processing in natural language processing (NLP). Here are some key contributions and themes from the cited works:

1. **WebGPT**: Explores browser-assisted question-answering systems enhanced by human feedback (Cobbe et al., 2021).

2. **Nearest Neighbor Language Models**: Investigates how memorization can aid generalization in language models (Khandelwal et al., 2019).

3. **Long-Context Language Models**: Discusses the capabilities of long-context models in subsuming various retrieval and query mechanisms (Lee et al., 2024).

4. **Retrieval-Augmented Generation (RAG)**: Examines the integration of retrieval mechanisms with generation tasks for knowledge-intensive NLP applications (Lewis et al., 2020).

5. **Dense Passage Retrieval**: Focuses on techniques for open-domain question answering using dense passage retrieval methods (Karpukhin et al., 2020).

6. **Surveys on RAG and Long-Context Models**: Several papers provide comprehensive surveys on the effectiveness and challenges of RAG versus long-context models (Gao et al., 2023; Li et al., 2024).

7. **Utilization of Long Contexts**: Studies how language models leverage long contexts and the implications for their performance (Zhang et al., 2023; Zhang et al., 2023).

8. **Challenges and Comparisons**: Discusses the challenges faced by long-context models and RAG systems, and presents comparative analyses (Laban et al., 2024; Kirkovska & Seethepalli, 2024).

These references collectively contribute to the understanding of how language models can be enhanced through retrieval mechanisms and the importance of context length in processing and generating language.

## Document 8

The provided text appears to be a list of references from an academic paper, specifically focusing on recent research related to long-context natural language processing (NLP), financial question answering, and benchmarks for question answering research. Here are the key contributions and topics from the references:

1. **Long Context NLP**:
- Goldman et al. (2024) discuss the challenges of long-context NLP and propose a framework for evaluating the necessity of retrieval in long-context scenarios.

2. **Retrieval-Augmented Generation (RAG)**:
- Jin et al. (2024) address the integration of long-context large language models (LLMs) with RAG techniques, focusing on overcoming challenges associated with processing long inputs.

3. **Financial Question Answering**:
- Islam et al. (2023) introduce Financebench, a new benchmark designed specifically for evaluating financial question answering systems.

4. **Question Answering Benchmarks**:
- Kwiatkowski et al. (2019) present the Natural Questions benchmark, which has become a standard for assessing question answering capabilities in NLP.

5. **Evaluation of LLMs**:
- Zheng et al. (2023) explore the evaluation of LLMs using MT-Bench and Chatbot Arena, providing insights into their performance as evaluators.

These references highlight ongoing research efforts to improve NLP systems' capabilities in handling long contexts, financial queries, and the evaluation of language models.

## Document 9

### Appendix

#### A. Model Versions
The following model versions were benchmarked in this study:

| Model              | Release     | API Version                              | Max Context |
|--------------------|-------------|------------------------------------------|-------------|
| o1-mini            | 2024-9-12   | o1-mini-2024-09-12                       | 128k        |
| o1-preview         | 2024-9-12   | o1-preview-2024-09-12                    | 128k        |
| Gemini 1.5 Pro     | 2024-6-27   | gemini-1.5-pro-001                       | 2,000k      |
| Gemini 1.5 Flash   | 2024-6-27   | gemini-1.5-flash-001                     | 2,000k      |
| GPT-4o             | 2024-5-13   | gpt-4o-2024-05-13                        | 128k        |
| Claude 3.5 Sonnet  | 2024-6-20   | claude-3-5-sonnet-20240620               | 200k        |
| Claude 3 Opus      | 2024-2-29   | claude-3-opus-20240229                   | 200k        |
| Claude 3 Haiku     | 2024-3-14   | claude-3-haiku-20240307                  | 200k        |
| GPT-4o-mini        | 2024-7-18   | gpt-4o-mini-2024-07-18                   | 128k        |
| GPT-4-turbo        | 2024-04-09  | gpt-4-turbo-2024-04-09                   | 128k        |
| Claude 3 Sonnet    | 2024-02-29  | claude-3-sonnet-20240229                 | 200k        |
| GPT-4              | 2023-01-25  | gpt-4-0125-preview                       | 128k        |
| GPT-3.5-turbo      | 2023-01-25  | gpt-3.5-turbo-0125                       | 16k         |
| Llama 3.1 405B     | 2024-07-23  | meta-llama/Llama-3.1-405B-Instruct       | 128k        |
| Llama 3 70B        | 2024-03-18  | meta-llama/Meta-Llama-3-70B              | 8k          |
| Llama 3.1 70B      | 2024-07-23  | meta-llama/Llama-3.1-70B                 | 128k        |
| Llama 3.1 8B       | 2024-07-23  | meta-llama/Llama-3.1-8B-Instruct         | 128k        |
| Qwen-2-72B         | 2024-06-06  | Qwen/Qwen2-72B-Instruct                  | 128k        |
| Mixtral-8x7B       | 2023-12-11  | mixtral-8x7b-instruct-v0.1               | 32k         |
| DBRX               | 2024-3-27   | databricks/dbrx-instruct                 | 32k         |

*Table S1: LLMs evaluated in this study include closed source, API-based models (top) and open-source models (bottom).*

#### B. Dataset Details
The study benchmarked all LLMs on three curated RAG datasets formatted for both retrieval and generation:

| Dataset                 | Corpus     | Queries | Av. doc length (tokens) | Max doc length (tokens) |
|-------------------------|------------|---------|--------------------------|--------------------------|
| Databricks DocsQA       | 7,563      | 139     | 2856                     | 225,941                  |
| FinanceBench            | 53,399     | 150     | 811                      | 8,633                    |
| Natural Questions (dev split) | 7,369 | 534     | 11,354                   | 13,362                   |

*Table S2: Dataset details for the three datasets used in our end-to-end RAG benchmark.*

Individual answer correctness plots for Databricks DocsQA and Natural Questions are included in Figures S1 and S2. The performance of the Gemini 1.5 models evaluated on up to 2 million tokens can be found in Table S4.

## Document 10

The provided data presents the performance of various language models (LLMs) in terms of answer correctness across different token lengths. The models are evaluated on their ability to provide correct answers, with scores ranging from 0 to 1, where higher scores indicate better performance.

### Key Contributions:
1. **Model Performance Comparison**: The table lists multiple models, including "o1-preview-2024-09-12," "gpt-4o-2024-05-13," and "gemini-1.5-pro," among others, along with their average correctness scores and performance at various token lengths (2k to 2000k).
2. **Long Context Performance**: Additional tables (S3 and S4) provide insights into model performance specifically for long contexts, highlighting how models like "Gemini 1.5 Pro" and "Gemini 1.5 Flash" perform at higher token counts (up to 2000k).

### Methodology:
- The models were likely evaluated using a standardized dataset (DocsQA) to measure their answer correctness based on the context length provided.
- The correctness scores are calculated based on the models' responses to questions, assessing their ability to maintain accuracy as the context length increases.

### Datasets Used:
- The evaluation appears to utilize the DocsQA dataset, which is designed for assessing question-answering capabilities in long contexts.

### Key Results:
- The highest average correctness score is observed for "o1-preview-2024-09-12" at 0.763, while "gpt-3.5-turbo" has the lowest at 0.44.
- For long contexts, "Gemini 1.5 Pro" maintains a strong performance with scores around 0.633 at 256k tokens, while "Gemini 1.5 Flash" shows lower performance with scores around 0.522.

### Conclusion:
The data indicates significant variability in model performance based on context length, with some models demonstrating robust capabilities in handling longer contexts, while others struggle. This highlights the importance of model selection based on specific use cases, particularly in applications requiring extensive context comprehension.

## Document 11

The document discusses the performance of various models in answering questions based on long contexts, specifically focusing on retrieval-augmented generation (RAG) methods. Key contributions and findings include:

1. **Model Performance**: The paper presents a comparison of different models (e.g., GPT-4, Claude, Ilama) in terms of their answer correctness on long contexts, with performance metrics visualized in a figure (Figure S2). The models are evaluated across varying context lengths, indicating how their performance changes with the amount of context provided.

2. **Retrieval Performance**: The study assesses how the number of retrieved chunks impacts the recall score, which serves as an upper bound for the generation model's performance. The recall@k results for the OpenAI text-embedding-3-large model are provided across three datasets (Databricks DocsQA, FinanceBench, and NQ) and different context lengths, as shown in Table S5.

3. **Saturation Points**: The results indicate that each dataset reaches a saturation point for recall at different context lengths. For instance, the NQ dataset saturates at 8k tokens, while DocsQA and FinanceBench reach saturation at 96k and 128k tokens, respectively. This suggests that larger context sizes can capture more relevant information, enhancing the overall quality of the system.

4. **Implications for RAG**: The findings highlight that while retrieval accuracy improves with more context, this does not guarantee a corresponding increase in RAG accuracy, indicating a complex relationship between retrieval and generation performance.

Overall, the research emphasizes the importance of context length and retrieval strategies in enhancing the performance of models in natural question answering tasks.

## Document 12

The section discusses the evaluation methodology using the "LLM-as-a-judge" paradigm to assess the correctness of generated answers against ground truth answers. The evaluation framework employed is from Databricks, which has been calibrated with human preferences on datasets such as FinanceBench and Databricks DocsQA. The judge demonstrated a strong agreement with human labelers, achieving an agreement rate of 88.1 ± 5.5% and a Cohen’s kappa score of 0.64 ± 0.13.

The prompts used for evaluation across different datasets are outlined as follows:

1. **Databricks DocsQA**: The assistant is tasked with answering questions related to Databricks products or Spark features, using relevant passages provided as context.

2. **FinanceBench**: The assistant answers questions related to financial reports, again using relevant passages for context.

3. **Natural Questions (NQ)**: The assistant answers questions based on retrieved context, with an emphasis on providing concise answers.

Each prompt instructs the assistant to focus only on relevant passages and to utilize its knowledge if no relevant information is available.

## Document 13

The section discusses the failure modes of generation models when handling longer context lengths. The authors identified several categories of failures based on manual inspection of model outputs at varying context lengths. The defined categories include:

1. **repeated_content**: The model generates answers that consist entirely of repeated words or characters.
2. **random_content**: The output is completely random, irrelevant, or lacks logical and grammatical coherence.
3. **fail_follow_inst**: The model fails to understand or follow the instruction given in the prompt, such as misinterpreting a request for an answer based on context as a request for a summary.
4. **empty_resp**: The model produces no response at all.
5. **wrong_answer**: The model attempts to follow the instruction but provides an incorrect answer.
6. **others**: Any failure that does not fit the above categories, including:
- **refusal**: The model refuses to answer or claims the context is not relevant.
- **task_failed**: The model's API blocks the prompt due to filtering guidelines, which is not included in the final correctness assessment.

To classify these failures, the authors developed a prompt template that guides a model (GPT-4o) to categorize the failures based on the question, expected answer, and generated answer. They note that failure patterns may vary across different datasets and generation settings.

## Document 14

The section discusses the failures of the Claude 3 Sonnet model in responding to natural language questions, primarily due to its refusal to provide answers based on copyright concerns. It presents a table (Table S6) with examples of these failures, illustrating the discrepancy between expected answers and the generated responses.

In the examples provided:

1. **Question**: "Who played Mrs. Warboys in One Foot in the Grave?"
- **Expected Answer**: Doreen Mantle
- **Generated Answer**: The model refuses to provide the answer, citing copyright restrictions and offers to summarize or paraphrase instead.

2. **Question**: "When did Korn’s Follow the Leader come out?"
- **Expected Answer**: August 18, 1998
- **Generated Answer**: Again, the model declines to provide the answer due to copyright concerns, suggesting it can summarize related information instead.

3. **Question**: "Who plays Captain Phasma in Star Wars: The Force Awakens?"
- **Expected Answer**: Gwendoline Christie
- **Generated Answer**: The model refuses to quote copyrighted material and offers to provide a summary or personal thoughts instead.

These examples highlight a significant limitation of the Claude 3 Sonnet model in adhering to user instructions when the responses involve copyrighted content.

## Document 15

The section discusses the failures of various models, specifically GPT-4, Mixtral-instruct, and DBRX-instruct, when responding to questions from the Natural Questions dataset.

### GPT-4 Failures
- **Type of Errors**: GPT-4 often provides incorrect answers or irrelevant responses.
- **Examples**:
- For the question "who sang once upon a dream at the end of maleficent," the expected answer is "Lana Del Rey," but GPT-4 generated "Ariana Grande & John Legend."
- Another example is the question "who was elected president in Mexico in 2000," where GPT-4's response was unrelated: "15th largest in nominal terms and 11th largest by purchasing power parity."

### Mixtral-instruct and DBRX Failures
- **Mixtral-instruct**: This model frequently outputs repeated or irrelevant content. For instance, when asked "who wrote the book the origin of species," it generated repeated characters for "dream" in Chinese.
- **DBRX-instruct**: This model tends to summarize content instead of directly answering questions. An example includes the question "who was the top scorer in 2014 world cup," where the response was a lengthy summary of a table of top goalscorers rather than a direct answer.

### Summary of Findings
- The failures highlight the challenges these models face in understanding and accurately responding to specific queries, with GPT-4 making factual errors, while Mixtral and DBRX struggle with content relevance and adherence to instructions.

## Document 16

In the evaluation of Gemini 1.5 Pro on the Natural Questions (NQ) benchmark, two primary failure modes were identified: task_failed and wrong_answer. The task_failed instances were largely attributed to the strict content filtering implemented by the Gemini API, which was particularly evident with the NQ dataset. This filtering was observed to increase with the length of the context provided. An example of this filtering is illustrated by a BlockedPromptException, which indicated that the content was blocked due to safety concerns, including categories such as sexually explicit content and hate speech, despite the NQ dataset not being known for such content.

In contrast, other APIs like OpenAI and Anthropic did not exhibit similar filtering issues during benchmarking. It is important to note that queries resulting in task_failed due to filtering were excluded from the final accuracy score. Despite these challenges, Gemini 1.5 Pro, along with Flash, achieved high answer correctness values exceeding 0.85 when evaluated with a context length of 2 million tokens.

The second major reason for failure was wrong_answer, where the generated answers did not match the expected answers. Examples of this include incorrect responses to questions about historical figures and events, as shown in a provided table. Overall, while Gemini 1.5 Pro demonstrated strong performance in certain aspects, the issues with content filtering and incorrect answers highlight areas for improvement.

## Document 17

The section discusses the performance of the Gemini 1.5 Pro model on the Databricks DocsQA dataset, highlighting the nature of its failures. Unlike other datasets, the primary issue observed is not due to safety filtering but rather incorrect answers.

### Key Contributions:
- The analysis categorizes failures into several types, with a significant portion attributed to "wrong_answer."
- The model's performance is evaluated across various context lengths, revealing that as context length increases, the majority of failures remain in the "wrong_answer" category.

### Methodology:
- The evaluation involved analyzing the model's responses to specific questions and comparing them against expected answers.
- Examples of incorrect answers are provided to illustrate the model's shortcomings.

### Datasets Used:
- The focus is on the Databricks DocsQA dataset, which is designed to assess the model's ability to answer questions related to Databricks documentation.

### Key Results:
- The model's generated answers often lack critical details, as seen in the provided examples. For instance, while the generated answer about auto optimization in streaming deltas is mostly correct, it omits important features like optimized writes and auto compaction.

This analysis indicates that while the model performs reasonably well, there are notable gaps in the completeness of its answers, particularly in technical contexts.

## Document 18

The Databricks UI allows users to create a model serving endpoint by following these steps:

1. Click on **Serving** in the sidebar to open the Serving UI.
2. Click on **Create serving endpoint**.
3. Enter a name for your endpoint in the **Serving endpoint name** field.
4. In the **Edit configuration** section, select the model and its version that you wish to serve.
5. Choose the size of the compute resources for the endpoint.
6. Specify whether the endpoint should automatically scale to zero when not in use and set the percentage of traffic to route to the served model.
7. Click on **Create serving endpoint**.

Initially, the **Serving endpoint state** will display as Not Ready. After a few minutes, it will change to Ready once the endpoint is operational. Additionally, you can create an endpoint directly from the registered model page by selecting the model, clicking the **Use model for inference** button, and following similar steps to configure the endpoint.

## Document 19

The document presents examples of failures in AI models, specifically Gemini 1.5 Pro and Llama 3.1 405B, when responding to questions from the FinanceBench and Natural Questions datasets, respectively.

In the FinanceBench dataset, the model's generated answer regarding 3M's liquidity profile is deemed insufficient as it fails to provide the specific quick ratio for Q2 of FY2023, which is crucial for assessing liquidity. The expected answer indicates that the quick ratio was 0.96, suggesting a need for improvement to reach a healthy liquidity mark of 1x. The model does mention that 3M maintains a strong liquidity profile but does not directly answer the question.

In the Natural Questions dataset, the Llama 3.1 model provides incorrect answers to questions about the number of episodes in "Attack on Titan" and the first use of the chain in F1, indicating a lack of accuracy in its responses.

These examples highlight the challenges AI models face in providing precise and contextually relevant answers based on the information available.

## Document 20

The table presented outlines the costs associated with various API-based models for processing input tokens, specifically for maximum sequence lengths of 8k, 64k, 128k, and 2 million tokens. The cost values are current as of October 2024 and are expressed in dollars per million tokens.

Key points from the table include:

- **Models and Costs**: The models listed include GPT4o, GPT4o-mini, o1-preview, Claude 3.5 Sonnet, Claude 3 Opus, Claude 3.5 Haiku, Gemini 1.5 Pro, and Gemini 1.5 Flash. Each model has different costs associated with varying sequence lengths.

- **Cost Estimates**:
- For example, GPT4o has a cost of $2.5 per million tokens, with specific costs for 128k tokens being $0.32 per query, leading to a total estimated cost (Cost A) of $263.36 for 823 queries.
- In contrast, the Gemini 1.5 Pro model has a cost of $1.25 per million tokens, with a total estimated cost (Cost B) of $4115 for 823 queries at a maximum sequence length of 2 million tokens.

- **Full Benchmarking Costs**: The table also provides estimated costs for "full benchmarking" across three datasets, highlighting the financial implications of using these models for extensive queries.

This information is crucial for understanding the economic considerations when selecting models for tasks requiring long context retrieval-augmented generation (RAG).

