# QA dataset template example
# used to convert original QA data to instruction tuning format

TEMPLATE = {
    "system": "You are a helpful assistant that answers questions accurately and concisely.",
    "instruction": "Please answer the following question based on the given context.",
    "input": "Question: {sample.question}\n\nContext: {sample.context}",
    "output": "{sample.answer}"
}

# more complex template example:
# TEMPLATE = {
#     "system": "You are an expert in answering questions. Please provide accurate and helpful responses.",
#     "instruction": "Based on the provided context, answer the question as accurately as possible. If the context doesn't contain enough information, say so.",
#     "input": "Context: {sample.context}\n\nQuestion: {sample.question}",
#     "output": "{sample.answer}"
# }

# if the original data format is different, you can map it like this:
# TEMPLATE = {
#     "system": "You are a helpful assistant.",
#     "instruction": "Answer the question: {sample.Question}",
#     "input": "Search results: {sample.SearchResults}",
#     "output": "{sample.Answer.Value}"
# } 