import random
from collections import defaultdict, Counter

class MarkovChain:
    def __init__(self, order=2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.starts = []

    def train(self, tokens):
        if len(tokens) < self.order:
            return

        # Store possible starts (first 'order' tokens)
        self.starts.append(tuple(tokens[:self.order]))

        for i in range(len(tokens) - self.order):
            state = tuple(tokens[i:i+self.order])
            next_token = tokens[i+self.order]
            self.transitions[state][next_token] += 1

    def get_probability(self, state, next_token):
        """
        Returns P(next_token | state)
        state: tuple of tokens (length = order)
        """
        if state not in self.transitions:
            return 0.0
        
        total_count = sum(self.transitions[state].values())
        token_count = self.transitions[state][next_token]
        
        return token_count / total_count if total_count > 0 else 0.0

    def generate(self, length=50):
        if not self.starts:
            return ""

        current_state = random.choice(self.starts)
        result = list(current_state)

        for _ in range(length - self.order):
            if current_state not in self.transitions:
                break

            possible_next = self.transitions[current_state]
            total = sum(possible_next.values())
            
            # Weighted random choice
            choices = list(possible_next.keys())
            weights = list(possible_next.values())
            
            next_token = random.choices(choices, weights=weights, k=1)[0]
            
            result.append(next_token)
            current_state = tuple(result[-self.order:])

        return " ".join(result)

if __name__ == "__main__":
    # Simple test
    tokens = "a b c a b c a b d a b e".split()
    mc = MarkovChain(order=2)
    mc.train(tokens)
    print(f"Transitions: {dict(mc.transitions)}")
    print(f"Generated: {mc.generate(10)}")
