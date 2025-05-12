import streamlit as st
from mcp_server import agent_executor

st.title("Business Continuity Insurance Model Analysis")
st.write("Ask questions about how parameters affect the model, or request specific analysis.")

with st.sidebar:
    st.header("About")
    st.write("""
    This tool uses Model Context Protocol (MCP) with Together AI's Llama 3 
    to analyze the Business Continuity Insurance simulation model.
    """)

query = st.text_area("Enter your question:", height=100)
run_button = st.button("Run Analysis")

if run_button and query:
    with st.spinner("Running analysis..."):
        try:
            response = agent_executor.invoke({"input": query})
            
            st.subheader("Analysis Results")
            st.write(response["output"])
            
            # Check if there are any charts to display
            if "charts" in response:
                for chart in response["charts"]:
                    st.plotly_chart(chart)
        except Exception as e:
            st.error(f"Error during analysis: {e}")