# bert-large-legal-sentence-classification
### Check [HuggingFace repository](https://huggingface.co/samkas125/bert-large-legal-sentence-classification) to clone and test the model.

## Model Description

[**bert-large-legal-sentence-classification**](https://huggingface.co/samkas125/bert-large-legal-sentence-classification) is a finetuned `bert-large-cased` model that is ready to use for legal sentence classification into rhetorical roles. It was fine-tuned using a repository of [analyzed disability-claim decisions](https://github.com/vernrwalker/VetClaims-JSON/) issued by the Board of Veterans' Appeals ("BVA") of the U.S. Department of Veterans Affairs.

### How to use

#### With HuggingFace pipeline

```python
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("samkas125/bert-large-legal-sentence-classification")
model = BertForSequenceClassification.from_pretrained('samkas125/bert-large-legal-sentence-classification')
nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)
example = "The Veteran did not have a psychiatric disorder in service that was unrelated to the use of drugs." 
results = nlp(example)
print(results)
```

#### Without HuggingFace pipeline

```python
from transformers import AutoTokenizer, BertForSequenceClassification
from torch import softmax, argmax
tokenizer = AutoTokenizer.from_pretrained("samkas125/bert-large-legal-sentence-classification")
model = BertForSequenceClassification.from_pretrained('samkas125/bert-large-legal-sentence-classification')
sentence = "The Veteran did not have a psychiatric disorder in service that was unrelated to the use of drugs."
encoded_input = tokenizer(sentence, return_tensors='pt')
output = model(**encoded_input)
logits = output.logits
probs = softmax(logits, dim=1)
predicted_class = argmax(probs, dim=1).item()
print(predicted_class)
```
Check the `config.json` file to map the index of a class to its name.

### Limitations and bias

This model is limited by the size of its training set of 6153 sentences. Additionally, it was only trained on disability claim decisions issued by the U.S. Department of Veterans Affairs. It may not generalize well to non-legal sentences, or sentences from other types of cases within the legal domain.

## Training data

A repository of [analyzed disability-claim decisions](https://github.com/vernrwalker/VetClaims-JSON/) issued by the Board of Veterans' Appeals ("BVA") of the U.S. Department of Veterans Affairs was used to finetune the `bert-large-cased` model for sequence classification. It consisted of 6153 sentences from 50 cases. The sentences were classified by the Research Laboratory for Law, Logic and Technology (LLT Lab) at the Maurice A. Deane School of Law at Hofstra University, in New York. The analysis consists of classifying sentences in those decisions according to their rhetorical role (as defined within the same repository).
