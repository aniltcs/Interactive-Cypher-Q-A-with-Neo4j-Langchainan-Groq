# Interactive-Cypher-Q-A-with-Neo4j-Langchainan-Groq

Created an app that executes cyper query to fetch structured data(Nodes,Relationship,Properties) from Neo4j Aura Database

App Url: https://interactive-cypher-q-a-with-neo4j-langchainan-groq.streamlit.app/

Questions
-----------

What are Movie title?

Find all Person name?

Which actors played in the movie Casino?

Query: MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a:Person) RETURN a.name

List all the genres of the movie Jumanji

Query: MATCH (m:Movie {title: 'Jumanji'})-[:IN_GENRE]->(g:Genre) RETURN g.name
