from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, Pipeline, pipeline

class CustomTextClassificationPipeline(Pipeline):
    def __init__(self, model_path, ner_model_name="FacebookAI/xlm-roberta-large-finetuned-conll03-english", multi_label=True, device=None):
        # DipeshY/distilbert-finetuned-disaster-entity
        # Initialize model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Initialize NER model and tokenizer
        ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
        ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)

        # Set device
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # Set attributes
        self.ner_pipeline = pipeline(task="ner", model=ner_model, tokenizer=ner_tokenizer, device=0 if device == "cuda" else -1)
        # Call the parent constructor, passing the tokenizer and model
        super().__init__(model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
        self.labels = [
            "non_disaster",
            "disaster",
            "flood",
            "extreme_rain",
            "earthquake",
            "typhoon",
            "landslide",
            "tsunami",
            "volcano",
            "wildfire",
            "informative",
            "non_informative"
        ]
        self.multi_label = multi_label
        self.cache_model_inputs = None
        self.cache_inputs = None

    def _sanitize_parameters(self, **kwargs):
        # Handle any additional parameters for preprocess, forward, and postprocess
        preprocess_kwargs = {}
        postprocess_kwargs = {"threshold": kwargs.get("threshold", 0.5)}
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, inputs):
        # Tokenize inputs and move them to the specified device
        self.cache_inputs = inputs
        return self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)

    def _forward(self, model_inputs):
        # Run model inference in evaluation mode, using no_grad for efficiency
        self.model.eval()
        self.cache_model_inputs = model_inputs
        with torch.no_grad():
            outputs = self.model(**self.cache_model_inputs)
            logits = outputs.logits
            # Apply sigmoid for multi-label classification or softmax for multi-class
            if self.multi_label:
                probabilities = torch.sigmoid(logits)
            else:
                probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    def postprocess(self, model_outputs, threshold=0.5):
        # Convert model outputs into a label-score dictionary, filtering by threshold
        for prob in model_outputs:
            classification_results = {
                label: round(score, 5)
                for label, score in zip(self.labels, prob)
                #if score >= threshold
            }
            # Add default if no scores above threshold
            if not classification_results:
                result = {"No predictions above threshold": 0.0}

            if (classification_results.get("disaster", 0) > threshold and
                    classification_results.get("informative", 0) > threshold and
                    classification_results.get("disaster", 0) > classification_results.get("non_disaster", 0)):
                # If condition met, perform NER

                #self.cache_model_inputs['input_ids'][0][0] = 6
                ner_results = {"text": self.cache_inputs,
                               "entities": self.ner_pipeline.predict(self.cache_inputs)}
                results = {"classification": classification_results, "ner_results": ner_results}
            else:
                results = {"classification": classification_results, "ner_results": None}

        return results
class DisasterClassifier:
    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.MULTI_LABEL = True

        self.LABELS = [
            "non_disaster",
            "disaster",
            "flood",
            "extreme_rain",
            "earthquake",
            "typhoon",
            "landslide",
            "tsunami",
            "volcano",
            "wildfire",
            "informative",
            "non_informative"
        ]

    def predict(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probabilities = torch.sigmoid(logits).cpu().numpy() if self.MULTI_LABEL else torch.softmax(logits, dim=1).cpu().numpy()

            results = []
            for prob in probabilities:
                result = sorted(zip(self.LABELS, prob), key=lambda x: x[1], reverse=True)
                results.append(result)

        return {"predictions": results}

# If you want to run this file directly for testing, you can include this part
if __name__ == "__main__":
    model_path = Path("./model")  # Adjust as necessary
    classifier2 = DisasterClassifier(model_path)
    classifier = CustomTextClassificationPipeline(model_path)
    classifier3 = pipeline("ner", model="FacebookAI/xlm-roberta-large-finetuned-conll03-english")
    sample_text = "Moderate magnitude 5.6  #Earthquake 52 miles southeast of Chiba, Japan.  10:07pm at epicenter (1h ago, 10km deep) https://t.co/1v5v9zv3Zw"
    print(classifier.predict(sample_text))
    print(classifier2.predict(sample_text))
    print(classifier3.predict(sample_text))
    print(classifier3.predict(sample_text))
    print(classifier3.predict("Fukushima nuclear disaster evacuation zone now teems with wildlife https://t.co/IRYE3KHP5u via"))
    tokenized_input = classifier3.tokenizer(sample_text, return_tensors="pt")
    # Tokenize the input text
    token_ids = tokenized_input["input_ids"][0]  # Use [0] to get the tensor as a list
    token_ids_list = token_ids.tolist()  # Convert to a regular list
    tokenizer3 = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-english")
    vocab_by_id = sorted(tokenizer3.vocab.items(), key=lambda item: item[1])
    print("First 20 tokens in vocab:", vocab_by_id[:20])


    # Print token IDs and corresponding tokens
    print("Token IDs:", token_ids_list)
    print("Result:", classifier3.tokenizer.decode(token_ids.to(torch.device("cuda"))))