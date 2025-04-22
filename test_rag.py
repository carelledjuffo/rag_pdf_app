from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

def eval_retrieval_precision(input, actual_output, expected_output, retrieval_context):

    metric = ContextualPrecisionMetric(
        threshold=0.7,
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
    print("################################################")
    print("Score = ", metric.score)
    print("Reason = ", metric.reason)
    print("################################################")

    #evaluate(test_cases=[test_case], metrics=[metric])
    #return {"score": metric.score, "reason": metric.reason}