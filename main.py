import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from markov_chain import MarkovChain
from evaluator import Evaluator

RESULTS_DIR = "results"
REPORT_DIR = "report"

def ensure_dirs():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

def main():
    ensure_dirs()
    
    # 1. Load Data
    print("Loading data...")
    loader = DataLoader()
    poems = loader.get_poems()
    
    all_tokens = []
    for poem in poems:
        all_tokens.extend(loader.clean_text(poem))
        
    print(f"Total tokens: {len(all_tokens)}")
    
    # 2. Train Models
    print("Training models...")
    mc2 = MarkovChain(order=2)
    mc3 = MarkovChain(order=3)
    
    for poem in poems:
        tokens = loader.clean_text(poem)
        if not tokens:
            continue
        mc2.train(tokens)
        mc3.train(tokens)
    
    # 3. Generate and Evaluate
    print("Generating and evaluating...")
    evaluator = Evaluator()
    
    results = {
        "order_2": {"texts": [], "metrics": {"perplexity": [], "repetition_rate": []}},
        "order_3": {"texts": [], "metrics": {"perplexity": [], "repetition_rate": []}}
    }
    
    # Generate 10 texts for each model
    for order, model, key in [(2, mc2, "order_2"), (3, mc3, "order_3")]:
        print(f"Processing Order {order}...")
        for i in range(10):
            text = model.generate(length=30)
            results[key]["texts"].append(text)
            
            pp = evaluator.calculate_perplexity(model, text)
            rr = evaluator.repetition_rate(text)
            
            results[key]["metrics"]["perplexity"].append(pp)
            results[key]["metrics"]["repetition_rate"].append(rr)
            
        # POS Distribution Distance (comparing all generated texts to original corpus)
        # We need to reconstruct original text from tokens or use raw poems
        # Let's use raw poems for original reference
        pos_dist = evaluator.pos_distribution_distance(poems, results[key]["texts"])
        results[key]["metrics"]["pos_dist_distance"] = pos_dist
        
    # 4. Save Results
    with open(os.path.join(RESULTS_DIR, "generation_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    # 5. Visualize
    print("Generating visualizations...")
    
    # Comparison of Metrics
    metrics = ["perplexity", "repetition_rate"]
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        data = [
            results["order_2"]["metrics"][metric],
            results["order_3"]["metrics"][metric]
        ]
        # Filter out infs for plotting if any
        data_clean = [[x for x in d if x != float('inf')] for d in data]
        
        plt.boxplot(data_clean, labels=["Order 2", "Order 3"])
        plt.title(f"Comparison of {metric.replace('_', ' ').title()}")
        plt.ylabel(metric)
        plt.savefig(os.path.join(REPORT_DIR, f"comparison_{metric}.png"))
        plt.close()
        
    # POS Distance Bar Chart
    plt.figure(figsize=(8, 6))
    pos_dists = [
        results["order_2"]["metrics"]["pos_dist_distance"],
        results["order_3"]["metrics"]["pos_dist_distance"]
    ]
    plt.bar(["Order 2", "Order 3"], pos_dists, color=['skyblue', 'salmon'])
    plt.title("POS Distribution Distance (lower is better)")
    plt.ylabel("Jensen-Shannon Distance")
    plt.savefig(os.path.join(REPORT_DIR, "comparison_pos_dist.png"))
    plt.close()
    
    print("Done! Results saved to results/ and report/")

if __name__ == "__main__":
    main()
