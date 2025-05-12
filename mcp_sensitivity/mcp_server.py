from langchain_together import ChatTogether
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os
import logging
from mcp_sensitivity_analysis import mcp_sensitivity_analysis_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('mcp_server.log'),
        logging.StreamHandler()
    ]
)

# Initialize Together AI model
llm = ChatTogether(
    together_api_key="tgp_v1_JPoIO5MYozMmOp0lDZe4akswhlppaUtIsXKpIwygOkw",
    model="meta-llama/Llama-3-70b-chat-hf"
    #model = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
)

# Initialize document retrieval system
def create_vector_db():
    # Import Document class from langchain
    from langchain.schema import Document
    
    # Load documentation files
    documents = []
    for filename in ["iati_api.py", "hdx_integration.py", "integrated_model.py", 
                    "integrated_app.py", "transparent_viz.py", "sensitivity_analysis.py",
                    "generate_performance_matrix.py"]:
        if os.path.exists(filename):
            try:
                # Explicitly specify UTF-8 encoding
                with open(filename, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Create proper LangChain Document object
                    documents.append(Document(page_content=content, metadata={"source": filename}))
            except UnicodeDecodeError:
                # Fallback to Latin-1 which can decode any byte
                with open(filename, "r", encoding="latin-1") as f:
                    content = f.read()
                    # Create proper LangChain Document object
                    documents.append(Document(page_content=content, metadata={"source": filename}))
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()  # You might want to use Together AI embeddings if available
    vector_store = FAISS.from_documents(docs, embeddings)
    
    return vector_store
    

# Create vector database for retrieval
docsearch = create_vector_db()

# Set up retrieval QA system
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

# New function for performance matrix generation
def generate_and_format_performance_matrix():
    try:
        from generate_performance_matrix import generate_performance_matrix
        
        # Generate the matrix
        performance_df = generate_performance_matrix()
        
        # Convert to dictionary format for JSON serialization
        result = {
            "matrix": performance_df.to_dict(orient="records"),
            "metrics": performance_df.columns.tolist(),
            "configurations": performance_df["Configuration"].tolist()
        }
        
        # Add summary insights
        result["insights"] = {
            "best_overall_config": performance_df.iloc[performance_df["Avg Organization Balance"].idxmax()]["Configuration"],
            "most_profitable_config": performance_df.iloc[performance_df["Insurer Profit"].idxmax()]["Configuration"],
            "most_stable_config": performance_df.iloc[performance_df["Balance Volatility"].idxmin()]["Configuration"],
            "improvements": {
                "balance": performance_df.iloc[-1]["Avg Organization Balance Improvement"],
                "volatility": performance_df.iloc[-1]["Balance Volatility Improvement"],
                "profit": performance_df.iloc[-1]["Insurer Profit Improvement"]
            }
        }
        
        return result
    except Exception as e:
        logging.error(f"Error generating performance matrix: {e}")
        return {"error": str(e)}

# Define the tools that the agent can use
tools = [
    Tool(
        name="SensitivityAnalysis",
        func=lambda x: json.dumps(mcp_sensitivity_analysis_api(x)), # Remove json.loads
        description="Run sensitivity analysis on the Business Continuity Insurance Model. Input should be a JSON object with 'params_to_vary' (dict mapping parameter names to min/max ranges), 'metrics' (list of metrics to analyze), and optional 'sim_duration'. Example: {\"params_to_vary\": {\"claim_trigger\": [0.1, 0.9]}, \"metrics\": [\"Number of Claims\", \"Insurer Profit\"], \"sim_duration\": 12}"
    ),
    Tool(
        name="NaturalLanguageAnalysis",
        func=lambda x: json.dumps(mcp_sensitivity_analysis_api({"query": x})),
        description="Analyze how changing parameters affects the Business Continuity Insurance Model using natural language. Input should be a question about how parameters affect outcomes."
    ),
    Tool(
        name="DocumentRetrieval",
        func=lambda x: qa.run(x),
        description="Retrieve information about the Business Continuity Insurance Model from documentation. Input should be a question about how the model works, parameters, or implementation details."
    ),
    Tool(
        name="GeneratePerformanceMatrix",
        func=lambda x: json.dumps(generate_and_format_performance_matrix()),
        description="Generate a performance comparison matrix that analyzes how different data sources (HDX/IATI) affect model outcomes across multiple metrics. This tool runs multiple simulations to compare baseline, IATI-only, HDX-only, and fully integrated configurations. No input is needed - just call the tool."
    )
]

# Create the agent with Together AI's Llama 3 model
prompt = PromptTemplate.from_template("""
You are an expert in business continuity insurance modeling. Your goal is to help users understand how different parameters affect model outcomes.

The simulation analyzes how insurance protects humanitarian organizations in high-risk environments by considering outcomes such as:
- Insurer Profit: How profitable the insurance provider is.
- Average Organization Balance: The financial stability of humanitarian organizations.
- Number of Claims: The frequency of insurance claims.
- Risk Events: The occurrences of emergency and security incidents.

Key parameters include:
- premium_rate: Percentage of the organization's budget paid as premium (0.5-5%).
- payout_cap_multiple: Maximum payout as a multiple of the premium (1-5x).
- claim_trigger: Balance threshold for triggering an insurance claim (0.1-0.9).
- emergency_probability: Monthly likelihood of emergencies (1-15%).
- security_risk_factor: Monthly likelihood of security incidents (5-40%).
- waiting_period: Number of months required between claims (1-6).

You can also generate a performance matrix to compare how different data integration strategies affect model outcomes across multiple metrics. This matrix provides insights on how using HDX data, IATI data, or both together affects the model's performance compared to a baseline.

Your output should follow these rules:
1. **Direct Final Answers:**  
   - When you have a complete, final answer, output it as:  
     `Final Answer: <your answer>`  
   - Do not include any action commands when you provide the final answer.
2. **Using Tools:**  
   - If additional analysis is needed, use one of the following tools only: [SensitivityAnalysis, NaturalLanguageAnalysis, DocumentRetrieval, GeneratePerformanceMatrix].  
   - When invoking a tool, use the exact format:  
     ```
     Action: <Tool Name>
     Action Input: <Input>
     ```
3. **Chain-of-Thought:**  
   - Use your internal scratchpad for reasoning, but once your final answer is complete, do not include further tool calls.
4. **No Further Iterations:**  
   - Once you have output your final answer, end the process and do not continue iterating.

The tools available are:
{tool_names}

Tool details:
{tools}

Current scratchpad:
{agent_scratchpad}

Human: {input}
""")

# Create the agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15,  # Increase iterations
    max_execution_time=60  # Increase execution time in seconds (if supported)
)

# CLI interface for testing
if __name__ == "__main__":
    print("Business Continuity Insurance MCP Server")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break
        
        try:
            response = agent_executor.invoke({"input": query})
            print("\nResponse:", response["output"])
        except Exception as e:
            print(f"Error: {e}")