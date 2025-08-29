# Overview

This is a Business Continuity Insurance Model that simulates insurance dynamics for humanitarian organizations using real-world data sources. The project combines agent-based modeling with external APIs to create realistic simulations of how organizations manage financial risks during emergencies. The application provides both an interactive Streamlit interface and an AI-powered analysis tool using the Model Context Protocol (MCP) for intelligent querying and sensitivity analysis.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses **Streamlit** as the primary web framework, providing an interactive dashboard for running simulations and visualizing results. The frontend is organized into several key components:

- **Main Application Interface** (`integrated_app.py`) - Central dashboard for running simulations
- **Transparent Visualizations** (`transparent_viz.py`) - Custom Plotly-based charts and gauges for explaining model decisions
- **MCP Analysis Interface** (`mcp_streamlit.py`) - AI-powered query interface for advanced analysis

The choice of Streamlit enables rapid prototyping while providing professional-looking visualizations without complex frontend development.

## Backend Architecture
The backend follows a **modular microservices-like architecture** with clear separation of concerns:

### Core Simulation Engine
- **Agent-Based Model** (`integrated_model.py`) - Uses Mesa framework for simulating organization behaviors and insurance dynamics
- **Data Integration Layer** - Combines multiple external data sources into unified parameters

### Data Source Modules
- **HDX Integration** (`hdx_integration.py`) - Fetches security and emergency risk data from Humanitarian Data Exchange APIs
- **IATI Integration** (`iati_api.py`) - Retrieves organization financial profiles from International Aid Transparency Initiative

### AI Analysis Layer
- **MCP Server** (`mcp_server.py`) - Implements Model Context Protocol for LLM-powered analysis
- **Sensitivity Analysis** (`mcp_sensitivity_analysis.py`) - Automated parameter sensitivity testing

## Data Storage Solutions
The system uses a **hybrid caching strategy**:

- **File-based JSON caching** for API responses with 7-day expiry periods
- **In-memory session state** for Streamlit user interactions and simulation results
- **No persistent database** - all data is either cached or fetched on-demand

This approach reduces API calls while maintaining data freshness and avoids database complexity for a research-oriented application.

## Fallback and Resilience Design
The architecture implements **graceful degradation**:

- Primary mode uses live API data from HDX and IATI
- Automatic fallback to simulated data when APIs are unavailable
- Comprehensive error handling and logging throughout all modules
- Cache validation to handle network interruptions

## AI Integration Architecture
The system integrates **Together AI's Llama 3 model** through LangChain:

- **RAG (Retrieval-Augmented Generation)** using FAISS vector database for code documentation
- **Agent-based query processing** for complex analysis requests
- **Tool integration** allowing the AI to execute sensitivity analyses programmatically

# External Dependencies

## Core Data Sources
- **Humanitarian Data Exchange (HDX)** - ACAPS INFORM Severity Index for emergency parameters and HAPI/ACLED for security data
- **International Aid Transparency Initiative (IATI)** - Organization financial profiles and aid flow data

## AI and ML Services
- **Together AI** - LLM services using Llama 3-70B model for intelligent analysis
- **OpenAI** - Embeddings for document retrieval (optional dependency)

## Visualization and Analysis Libraries
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations and dashboards
- **Mesa** - Agent-based modeling framework (version 0.9.0 specifically)
- **Pandas/NumPy** - Data manipulation and analysis

## Development and Deployment
- **Python-dotenv** - Environment variable management
- **Docker/Dev Containers** - Containerized development environment
- **FAISS** - Vector similarity search for document retrieval

## API Authentication
- **ACAPS API** - Uses token-based authentication for emergency data
- **HAPI** - Application identifier-based access for humanitarian data
- **IATI** - Public API access with organization-specific endpoints

The system is designed to work offline with simulated data when external services are unavailable, making it resilient for research and demonstration purposes.