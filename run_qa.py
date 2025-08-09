# run_qa.py
from app.components.retriever import create_qa_chain

def main():
    # Load QA chain
    qa_chain = create_qa_chain()
    if qa_chain is None:
        print("❌ Failed to create QA chain. Check your vector store and LLM setup.")
        return

    print("✅ RAG Medical Bot ready. Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ["exit", "quit"]:
            break

        try:
            response = qa_chain.invoke({"query": question})
            answer = response.get("result", "No response found")
            print(f"💬 Answer: {answer}")
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()