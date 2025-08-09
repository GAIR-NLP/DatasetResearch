TEMPLATE_MULTI = {
    # specify the field name containing multiple questions
    "questions_field": "Questions",
    
    # template for each question-answer pair
    "template": {
        "instruction": "Answer the question: {sample.Question}",
        "input": "Context: {sample.Context}\\nDocument ID: {sample.DocID}",  # can reference shared fields
        "output": "{sample.Answer}"
    }
}