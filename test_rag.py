from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRelevancyMetric


# Evaluates if nodes in your retrieval_context that are relevant to the given input are ranked higher than irrelevant ones.
def eval_retrieval_precision(input, actual_output, expected_output, retrieval_context):

    metric = ContextualPrecisionMetric(
        include_reason=True
    )
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    # To run metric as a standalone
    metric.measure(test_case)
    print("################## Contextual Precision Metric ##############################")
    print("Score = ", metric.score)
    print("Reason = ", metric.reason)

# evaluates the overall relevance of the information presented in your retrieval_context
def eval_retrieval_relevance(input, actual_output, expected_output, retrieval_context):
    metric = ContextualRelevancyMetric(
        include_reason=True
    )
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    
    metric.measure(test_case)
    print("################# Contextual Relevancy Metric###############################")
    print("Score = ", metric.score)
    print("Reason = ", metric.reason)
    print("################################################")
