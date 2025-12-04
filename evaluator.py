import math
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
import spacy

class Evaluator:
    def __init__(self):
        try:
            self.nlp = spacy.load("pl_core_news_sm")
        except OSError:
            from spacy.cli import download
            download("pl_core_news_sm")
            self.nlp = spacy.load("pl_core_news_sm")

    def calculate_perplexity(self, model, text):
        """
        Calculates perplexity of the text given the model.
        PP(W) = exp(-1/N * sum(log P(wi | context)))
        """
        tokens = text.split() # Assuming text is space-separated tokens
        if len(tokens) < model.order + 1:
            return float('inf')
            
        log_prob_sum = 0
        N = len(tokens)
        count = 0
        
        for i in range(model.order, N):
            context = tuple(tokens[i-model.order : i])
            word = tokens[i]
            prob = model.get_probability(context, word)
            
            if prob > 0:
                log_prob_sum += math.log(prob)
            else:
                # Handle zero probability (smoothing or penalty)
                # For this assignment, we can assign a very small probability or skip
                # But skipping makes perplexity artificially low. 
                # Let's assign a small epsilon.
                log_prob_sum += math.log(1e-10)
            count += 1
            
        if count == 0:
            return float('inf')
            
        return math.exp(-log_prob_sum / count)

    def repetition_rate(self, text, n=3):
        tokens = text.split()
        if len(tokens) < n:
            return 0.0
            
        ngrams = [tuple(tokens[i : i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
            
        unique_ngrams = set(ngrams)
        return (1 - len(unique_ngrams) / len(ngrams)) * 100

    def get_pos_distribution(self, texts):
        pos_counts = Counter()
        for text in texts:
            doc = self.nlp(text)
            pos_counts.update([token.pos_ for token in doc])
            
        total = sum(pos_counts.values())
        if total == 0:
            return {}
            
        dist = {k: v / total for k, v in pos_counts.items()}
        return dist

    def pos_distribution_distance(self, original_texts, generated_texts):
        orig_dist = self.get_pos_distribution(original_texts)
        gen_dist = self.get_pos_distribution(generated_texts)
        
        # Align distributions
        all_pos = set(orig_dist.keys()) | set(gen_dist.keys())
        p = [orig_dist.get(pos, 0.0) for pos in all_pos]
        q = [gen_dist.get(pos, 0.0) for pos in all_pos]
        
        return jensenshannon(p, q)

if __name__ == "__main__":
    # Test
    evaluator = Evaluator()
    print("Evaluator initialized.")
