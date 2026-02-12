from RAG_pneumonia_agent import retriever


# Test queries related to pneumonia and chest X-rays
test_queries = [
    "COVID-19 chest X-ray features",
    "bacterial pneumonia radiography characteristics",
    "viral pneumonia imaging findings",
    "normal chest X-ray appearance",
    "pneumonia diagnosis on X-rays"
]

print("=" * 80)
print("RETRIEVER TEST")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"Query {i}: {query}")
    print(f"{'='*80}")
    
    try:
        results = retriever.invoke(query)
        print(f"Number of results: {len(results)}\n")
        
        for j, doc in enumerate(results, 1):
            print(f"--- Result {j} ---")
            print(f"Content: {doc.page_content[:300]}...")
            if hasattr(doc, 'metadata'):
                print(f"Metadata: {doc.metadata}")
            print()
    except Exception as e:
        print(f"Error retrieving results: {e}\n")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
