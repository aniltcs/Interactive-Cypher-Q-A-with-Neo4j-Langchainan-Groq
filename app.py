import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# -----------------------------
# Environment variables
# -----------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

# -----------------------------
# Neo4j connection
# -----------------------------
graph=Neo4jGraph(url=os.getenv("NEO4J_URI"),username=os.getenv("NEO4J_USERNAME"),password=os.getenv("NEO4J_PASSWORD"),refresh_schema=False )

# -----------------------------
# LLM
# -----------------------------
groq_api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

# -----------------------------
# Few-shot examples
# -----------------------------
examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {{title: 'Casino'}})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Jumanji",
        "query": "MATCH (m:Movie {{title: 'Jumanji'}})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
]

example_prompt=PromptTemplate.from_template(
    "User input:{question}\n Cypher query:{query}"
)

prefix = """
You are a strict Cypher expert.

Follow these rules:
1. NEVER use SQL keywords like SELECT, GROUP BY, HAVING.
2. ALWAYS use Cypher syntax: MATCH, WHERE, RETURN, ORDER BY, LIMIT.
3. RETURN only Cypher code — no explanations, no comments.
4. Do not invent properties or labels not present in the schema.
5. Use pattern:
   MATCH ...
   WHERE ...
   RETURN ...
"""

prompt = FewShotPromptTemplate(
    examples=examples[:5],
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question"],
)

# -----------------------------
# GraphCypherQAChain
# -----------------------------
qa_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    cypher_prompt=prompt,
    verbose=True,
    allow_dangerous_requests=True
)

# -----------------------------
# Streamlit Multi-tab UI
# -----------------------------
st.set_page_config(page_title="Neo4j Graph Q&A", layout="wide")

tabs = st.tabs(["Home / Q&A","History"])

# -----------------------------
# Session state for multi-turn chat
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# 1️⃣ Home / Q&A Tab
# -----------------------------
with tabs[0]:
    st.header("Neo4j Graph Q&A")
    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_question:
            try:
                response = qa_chain.invoke(user_question)
                st.success(response['result'])

                # Save to history
                st.session_state.chat_history.append({"question": user_question, "answer": response['result']})
            except Exception as e:
                st.error(f"Error: {e}")

# -----------------------------
# 3️⃣ Top-K Results Tab
# -----------------------------
# with tabs[1]:
#     st.header("Execute Cypher Query")
#     cypher_query = st.text_area("Enter Cypher to fetch top results (RETURN limited nodes):")
#     if st.button("Fetch Results"):
#         if cypher_query:
#             try:
#                 results = graph.query(cypher_query)
#                 st.write(results)
#             except Exception as e:
#                 st.error(f"Error executing Cypher: {e}")

# -----------------------------
# 4️⃣ History Tab
# -----------------------------
with tabs[1]:
    st.header("Chat History")
    if st.session_state.chat_history:
        for idx, chat in enumerate(st.session_state.chat_history[::-1]):
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            st.markdown("---")
    else:
        st.info("No history yet.")