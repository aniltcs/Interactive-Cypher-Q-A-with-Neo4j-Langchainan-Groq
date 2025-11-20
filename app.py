import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from neo4j import GraphDatabase

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

# -----------------------------
# Initialize Neo4j driver
# -----------------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Safe query runner using sessions
def run_cypher(query):
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]

# Initialize Neo4jGraph and override query to use session
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS, refresh_schema=False)
graph.query = run_cypher  # override internal query method

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# -----------------------------
# Few-shot examples
# -----------------------------
examples = [
    {"question": "How many artists are there?", "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)"},
    {"question": "Which actors played in the movie Casino?", "query": "MATCH (m:Movie {{title: 'Casino'}})<-[:ACTED_IN]-(a) RETURN a.name"},
    {"question": "How many movies has Tom Hanks acted in?", "query": "MATCH (a:Person {{name: 'Tom Hanks'}})-[:ACTED_IN]->(m:Movie) RETURN count(m)"},
    {"question": "List all the genres of the movie Jumanji", "query": "MATCH (m:Movie {{title: 'Jumanji'}})-[:IN_GENRE]->(g:Genre) RETURN g.name"},
    {"question": "Which actors have worked in movies from both the comedy and action genres?", "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name"},
]

example_prompt = PromptTemplate.from_template("User input:{question}\n Cypher query:{query}")

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
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question"]
)

# -----------------------------
# Initialize GraphCypherQAChain
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
st.set_page_config(page_title="Neo4j Graph Q&A (Groq LLM)", layout="wide")
tabs = st.tabs(["Home / Q&A", "Raw Cypher", "Chat History"])

# -----------------------------
# Session state for chat history
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
                st.session_state.chat_history.append({"type": "Graph QA", "question": user_question, "answer": response['result']})
            except Exception as e:
                st.error(f"Error: {e}")

# -----------------------------
# 2️⃣ Raw Cypher Tab
# -----------------------------
with tabs[1]:
    st.header("Execute Raw Cypher Query")
    cypher_query = st.text_area("Enter Cypher query (RETURN limited nodes for safety):")
    if st.button("Run Query"):
        if cypher_query:
            try:
                results = run_cypher(cypher_query)
                if results:
                    st.write(results)
                else:
                    st.info("Query returned no results.")
            except Exception as e:
                st.error(f"Error executing Cypher: {e}")

# -----------------------------
# 3️⃣ Chat History Tab
# -----------------------------
with tabs[2]:
    st.header("Chat History")
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history[::-1]:
            st.markdown(f"**Type:** {chat['type']}")
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            st.markdown("---")
    else:
        st.info("No history yet.")
