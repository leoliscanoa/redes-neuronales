from langchain_community.llms.ollama import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import AgentType
from flask import Flask, jsonify, request

app = Flask(__name__)

# endpoint de pruebas
@app.route('/prompt', methods=['POST'])
def prompt():
    user_prompt = request.json.get('prompt')
    if not user_prompt:
        return jsonify({"error": "Prompt is required"}), 400
    try:
        llm = Ollama(base_url="http://localhost:11434",model = 'gemma2:2b')
        database = SQLDatabase.from_uri("postgresql://postgres:postgres@localhost:5432/chinook_serial")
        toolkit = SQLDatabaseToolkit(db=database,llm=llm)
        agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
        response = agent_executor.invoke({"input": user_prompt})
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)