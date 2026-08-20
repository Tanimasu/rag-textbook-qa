"""Labels shared by the packaged Web interface."""

from types import MappingProxyType

RAGAS_METRIC_LABELS = MappingProxyType(
    {
        "faithfulness": "忠实度",
        "answer_relevancy": "答案相关性",
        "context_precision": "上下文精确度",
        "context_recall": "上下文召回率",
    }
)
