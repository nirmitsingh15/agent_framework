import openai

class QueryEngine:
    def __init__(self, vector_store, model_name="gpt-4o-mini"):
        self.vector_store = vector_store
        self.model_name = model_name

    def answer_question(self, query):
        """
        Answers a question by retrieving relevant chunks and using GPT for generation.

        :param query: User question
        :return: Generated response
        """
        relevant_chunks = self.vector_store.search(query, top_k=5)
        context = "\n".join(relevant_chunks)
        prompt = f"""
        Use the following context to answer the question.
        - Prioritize word-to-word matches from the context for exact answers.
        - If the question cannot be answered at all from the context, reply with "Data Not Available".
        
        Context:
        {context}

        Question: {query}

        Answer:
        """


        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response["choices"][0]["message"]["content"].strip()
