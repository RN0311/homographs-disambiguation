import openai
import json
import pandas as pd
from typing import List, Dict, Tuple
import time
from tqdm import tqdm

class HomographDisambiguator:
    def __init__(self, api_key: str):
        """Initialize the disambiguator with OpenAI API key."""
        openai.api_key = api_key
        self.system_prompt = """You are a homograph disambiguation expert. Given a sentence containing a homograph, 
        identify the correct meaning of the homograph based on context. Provide your answer in JSON format with 
        'word' and 'meaning' as keys."""
    
    def create_prompt(self, sentence: str, target_word: str) -> str:
        """Create a prompt for the GPT model."""
        return """Please disambiguate the word '{target_word}' in the following sentence:
        
        Sentence: "{sentence}"
        
        Return your response in this JSON format:
        {{
            "word": "the homograph word",
            "meaning": "brief definition based on context",
            "confidence": "high/medium/low"
        }}
      """

    def get_disambiguation(self, sentence: str, target_word: str, 
                         max_retries: int = 3) -> Dict[str, str]:
        """Get disambiguation for a single instance."""
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.create_prompt(sentence, target_word)}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
            
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error after {max_retries} attempts: {str(e)}")
                    return {"word": target_word, "meaning": "ERROR", "confidence": "none"}
                time.sleep(1) 

    def batch_process(self, data: List[Tuple[str, str]], 
                     batch_size: int = 10) -> List[Dict[str, str]]:
        """Process multiple sentences in batches."""
        results = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            batch_results = []
            for sentence, word in batch:
                result = self.get_disambiguation(sentence, word)
                batch_results.append(result)
            results.extend(batch_results)
            time.sleep(1) 
        return results

class Evaluator:
    @staticmethod
    def calculate_metrics(predictions: List[Dict[str, str]], 
                         ground_truth: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        correct = 0
        total = len(predictions)
        confidence_levels = {"high": 0, "medium": 0, "low": 0}
        
        for pred, truth in zip(predictions, ground_truth):
            if pred["meaning"].lower() == truth.lower():
                correct += 1
            confidence_levels[pred["confidence"]] += 1
        
        metrics = {
            "accuracy": correct / total,
            "high_confidence_ratio": confidence_levels["high"] / total,
            "medium_confidence_ratio": confidence_levels["medium"] / total,
            "low_confidence_ratio": confidence_levels["low"] / total
        }
        return metrics

def main():
    disambiguator = HomographDisambiguator(OPENAI_API_KEY)

    test_data = [
        ("The bass was too loud at the concert.", "bass"),
        ("He caught a huge bass in the lake.", "bass"),
        ("The wind will blow the leaves away.", "wind"),
        ("Let me wind up the clock.", "wind")
    ]
    
    results = disambiguator.batch_process(test_data)
    
    ground_truth = [
        "musical instrument sound",
        "type of fish",
        "moving air",
        "to twist or turn"
    ]
    
    evaluator = Evaluator()
    metrics = evaluator.calculate_metrics(results, ground_truth)
    
    print("\nResults:")
    for i, (result, truth) in enumerate(zip(results, ground_truth)):
        print(f"\nExample {i+1}:")
        print(f"Input: {test_data[i][0]}")
        print(f"Predicted meaning: {result['meaning']}")
        print(f"Ground truth: {truth}")
        print(f"Confidence: {result['confidence']}")
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()
