import sys
from cag.dataset import get

def verify_samples(dataset_name, num_samples=3):
    """Verify dataset samples by displaying question-answer pairs with context."""
    print(f"\nVerifying {dataset_name} dataset:")
    print("-" * 80)
    
    # Get dataset with minimal samples to verify
    texts, qa_pairs = get(
        dataset_name=dataset_name,
        max_knowledge=num_samples,
        max_questions=1,  # 1 question per document for clarity
        split=None
    )
    
    # Convert iterator to list to avoid exhausting it
    qa_pairs = list(qa_pairs)
    
    for i, (text, (question, answer)) in enumerate(zip(texts, qa_pairs)):
        print(f"\nSample {i + 1}:")
        print("\nContext (truncated):")
        print(text[:500] + "..." if len(text) > 500 else text)
        print("\nQuestion:")
        print(question)
        print("\nAnswer:")
        print(answer)
        print("-" * 80)

if __name__ == "__main__":
    verify_samples("natural-questions-train") 