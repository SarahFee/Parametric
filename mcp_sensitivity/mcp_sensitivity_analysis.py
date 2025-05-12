import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import json
from integrated_model import IntegratedBusinessContinuityModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('mcp_sensitivity_analysis.log'),
        logging.StreamHandler()
    ]
)

class MCPSensitivityAnalysis:
    """MCP-enabled sensitivity analysis for the Business Continuity Insurance Model."""
    
    def __init__(self):
        """Initialize the sensitivity analysis engine."""
        self.param_options = {
            "premium_rate": {"name": "Premium Rate (%)", "min": 0.5, "max": 5.0, "default": 2.0, "step": 0.5},
            "payout_cap_multiple": {"name": "Payout Cap (x Premium)", "min": 1.0, "max": 5.0, "default": 3.0, "step": 0.5},
            "claim_trigger": {"name": "Claim Trigger", "min": 0.1, "max": 0.9, "default": 0.5, "step": 0.1},
            "emergency_probability": {"name": "Emergency Probability (%)", "min": 1.0, "max": 15.0, "default": 5.0, "step": 1.0},
            "security_risk_factor": {"name": "Security Risk Factor (%)", "min": 5.0, "max": 40.0, "default": 20.0, "step": 5.0},
            "waiting_period": {"name": "Waiting Period (months)", "min": 1, "max": 6, "default": 2, "step": 1}
        }
        
        # Default fixed parameters
        self.default_params = {param: config["default"] for param, config in self.param_options.items()}
        
    def run_mcp_sensitivity_analysis(self, params_to_vary, metrics, sim_duration=12, use_hdx_data=True, 
                                   use_iati_data=True, hdx_api_key=None, num_points=5):
        """
        Run sensitivity analysis specified via MCP.
        
        Args:
            params_to_vary: Dict mapping parameter names to (min, max) tuples
            metrics: List of metrics to analyze
            sim_duration: Duration of simulation in months
            use_hdx_data: Whether to use HDX data
            use_iati_data: Whether to use IATI data
            hdx_api_key: API key for HDX data
            num_points: Number of points to sample for each parameter
            
        Returns:
            Dict containing analysis results
        """
        # Validate parameters
        valid_params = {}
        for param, (min_val, max_val) in params_to_vary.items():
            if param not in self.param_options:
                logging.warning(f"Invalid parameter: {param}")
                continue
                
            # Ensure min <= max
            if min_val > max_val:
                min_val, max_val = max_val, min_val
                
            # Apply parameter bounds
            param_config = self.param_options[param]
            min_val = max(min_val, param_config["min"])
            max_val = min(max_val, param_config["max"])
            
            valid_params[param] = (min_val, max_val)
            
        # Validate metrics
        valid_metrics = [m for m in metrics if m in ["Insurer Profit", "Average Organization Balance", 
                                                   "Number of Claims", "Risk Events"]]
        
        if not valid_params or not valid_metrics:
            return {"error": "No valid parameters or metrics provided"}
            
        # Determine analysis type
        if len(valid_params) == 1:
            return self._run_single_param_analysis(
                list(valid_params.keys())[0], 
                list(valid_params.values())[0],
                valid_metrics, sim_duration, use_hdx_data, use_iati_data, hdx_api_key, num_points
            )
        else:
            # For simplicity, handle only first two parameters for multi-param analysis
            param_items = list(valid_params.items())
            return self._run_multi_param_analysis(
                param_items[:min(len(param_items), 2)],
                valid_metrics, sim_duration, use_hdx_data, use_iati_data, hdx_api_key, num_points
            )
    
    def _run_single_param_analysis(self, param, value_range, metrics, sim_duration, 
                                 use_hdx_data, use_iati_data, hdx_api_key, num_points):
        """Run sensitivity analysis for a single parameter."""
        param_min, param_max = value_range
        param_range = np.linspace(param_min, param_max, num_points)
        
        # Initialize results storage
        results = {
            "parameter": param,
            "parameter_values": param_range.tolist(),
            "metrics": {}
        }
        
        for metric in metrics:
            results["metrics"][metric] = []
        
        # Fixed parameters (everything except the one we're varying)
        fixed_params = self.default_params.copy()
        del fixed_params[param]
        
        # Run simulations for each parameter value
        for param_value in param_range:
            # Set the parameter value (handling percentage conversions)
            params = fixed_params.copy()
            if param in ["emergency_probability", "security_risk_factor"]:
                params[param] = param_value / 100
            else:
                params[param] = param_value
            
            # Run the model
            model = run_model_with_params(params, sim_duration, use_hdx_data, use_iati_data, hdx_api_key)
            
            if model:
                # Collect metrics
                if "Insurer Profit" in metrics:
                    results["metrics"]["Insurer Profit"].append(model.insurer.profits)
                
                if "Average Organization Balance" in metrics:
                    results["metrics"]["Average Organization Balance"].append(
                        sum(org.balance for org in model.orgs) / len(model.orgs)
                    )
                
                if "Number of Claims" in metrics:
                    claim_count = sum(1 for log in model.event_log if "made a claim" in log.lower())
                    results["metrics"]["Number of Claims"].append(claim_count)
                
                if "Risk Events" in metrics:
                    emergency_count = sum(1 for log in model.event_log if "emergency" in log.lower())
                    security_count = sum(1 for log in model.event_log if "security" in log.lower())
                    results["metrics"]["Risk Events"].append(emergency_count + security_count)
            else:
                # Handle simulation failure
                for metric in metrics:
                    results["metrics"][metric].append(None)
        
        # Add analysis metadata
        results["analysis_type"] = "single_parameter"
        results["correlations"] = {}
        
        # Calculate correlations
        for metric in metrics:
            metric_values = results["metrics"][metric]
            # Filter out None values
            param_vals = [p for i, p in enumerate(param_range) if metric_values[i] is not None]
            metric_vals = [m for m in metric_values if m is not None]
            
            if len(metric_vals) > 1:
                correlation = np.corrcoef(param_vals, metric_vals)[0, 1]
                results["correlations"][metric] = correlation
            else:
                results["correlations"][metric] = None
                
        return results
    
    def _run_multi_param_analysis(self, param_items, metrics, sim_duration, 
                                use_hdx_data, use_iati_data, hdx_api_key, num_points):
        """Run sensitivity analysis for multiple parameters."""
        # Extract parameters and ranges
        params = [item[0] for item in param_items]
        param_ranges = [item[1] for item in param_items]
        
        # Generate parameter values to test
        param_values = [np.linspace(min_val, max_val, num_points) for min_val, max_val in param_ranges]
        
        # Create meshgrid for all parameter combinations
        param_meshes = np.meshgrid(*param_values)
        
        # Initialize results storage
        results = {
            "parameters": params,
            "parameter_values": [vals.tolist() for vals in param_values],
            "metrics": {},
            "analysis_type": "multi_parameter"
        }
        
        for metric in metrics:
            results["metrics"][metric] = np.zeros([num_points] * len(params)).tolist()
        
        # Fixed parameters (everything except the ones we're varying)
        fixed_params = self.default_params.copy()
        for param in params:
            del fixed_params[param]
        
        # Run simulations for each parameter combination
        for idx in np.ndindex(tuple([num_points] * len(params))):
            # Set parameter values
            sim_params = fixed_params.copy()
            for i, param in enumerate(params):
                param_value = param_values[i][idx[i]]
                
                # Handle percentage conversion
                if param in ["emergency_probability", "security_risk_factor"]:
                    sim_params[param] = param_value / 100
                else:
                    sim_params[param] = param_value
            
            # Run the model
            model = run_model_with_params(sim_params, sim_duration, use_hdx_data, use_iati_data, hdx_api_key)
            
            if model:
                # Collect metrics
                for metric in metrics:
                    if metric == "Insurer Profit":
                        metric_value = model.insurer.profits
                    elif metric == "Average Organization Balance":
                        metric_value = sum(org.balance for org in model.orgs) / len(model.orgs)
                    elif metric == "Number of Claims":
                        metric_value = sum(1 for log in model.event_log if "made a claim" in log.lower())
                    elif metric == "Risk Events":
                        emergency_count = sum(1 for log in model.event_log if "emergency" in log.lower())
                        security_count = sum(1 for log in model.event_log if "security" in log.lower())
                        metric_value = emergency_count + security_count
                    
                    # Store in nested result structure
                    result_ref = results["metrics"][metric]
                    for i in range(len(idx) - 1):
                        result_ref = result_ref[idx[i]]
                    result_ref[idx[-1]] = metric_value
            else:
                # Handle simulation failure
                for metric in metrics:
                    result_ref = results["metrics"][metric]
                    for i in range(len(idx) - 1):
                        result_ref = result_ref[idx[i]]
                    result_ref[idx[-1]] = None
        
        return results
   
    def natural_language_analysis(self, query, sim_duration=12, use_hdx_data=True, 
                                use_iati_data=True, hdx_api_key=None):
        """
        Process a natural language query for sensitivity analysis.
        
        Args:
            query: Natural language query about parameter sensitivity
            sim_duration: Duration of simulation in months
            use_hdx_data: Whether to use HDX data
            use_iati_data: Whether to use IATI data
            hdx_api_key: API key for HDX data
            
        Returns:
            Dict containing analysis results and interpretation
        """
        # This would be handled by the MCP server to parse the query
        # For this example, we'll just return a placeholder that explains
        # what would happen in a real MCP implementation
        
        return {
            "mcp_response": {
                "query_analysis": "In a real MCP implementation, this function would parse the natural language query to identify parameters and metrics of interest.",
                "execution_plan": "The MCP would then create an execution plan to run the appropriate sensitivity analysis.",
                "suggested_code": """
# Example code that MCP would generate and execute
analyzer = MCPSensitivityAnalysis()
results = analyzer.run_mcp_sensitivity_analysis(
    params_to_vary={"premium_rate": (1.0, 3.0), "claim_trigger": (0.3, 0.7)},
    metrics=["Insurer Profit", "Average Organization Balance"],
    sim_duration=12,
    use_hdx_data=True,
    use_iati_data=True,
    num_points=5
)
"""
            }
        }

# Helper function from your original code
def run_model_with_params(params, sim_duration, use_hdx_data, use_iati_data, hdx_api_key):
    """Helper function to run the model with specified parameters."""
    try:
        # Create and initialize model
        model = IntegratedBusinessContinuityModel(
            sim_duration=sim_duration,
            waiting_period=params.get('waiting_period', 2),
            payout_cap_multiple=params.get('payout_cap_multiple', 3.0),
            premium_rate=params.get('premium_rate', 2.0),
            custom_claim_trigger=params.get('claim_trigger', 0.5),
            emergency_probability=params.get('emergency_probability', 0.05),
            security_risk_factor=params.get('security_risk_factor', 0.2),
            use_hdx_data=use_hdx_data,
            use_iati_data=use_iati_data,
            hdx_api_key=hdx_api_key
        )
        
        # Initialize history lists if not already set
        if not hasattr(model, 'history'):
            model.history = []
        if not hasattr(model, 'profit_history'):
            model.profit_history = []
        if not hasattr(model, 'org_balance_history'):
            model.org_balance_history = {org.name: [] for org in model.orgs}

        # Run the simulation
        for step in range(sim_duration):
            model.step()
        
        return model
    except Exception as e:
        logging.error(f"Error running model: {e}")
        return None

# API endpoint for MCP server
def mcp_sensitivity_analysis_api(request_data):
    try:
        # Parse string to JSON if needed
        if isinstance(request_data, str):
            try:
                request_data = json.loads(request_data)
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON input: {e}")
                return {"error": f"Invalid JSON format: {e}. Please provide a valid JSON object."}
        
        # Rest of the function remains the same...
            
        analyzer = MCPSensitivityAnalysis()
        
        # Check if this is a natural language query
        if "query" in request_data:
            return analyzer.natural_language_analysis(
                query=request_data["query"],
                sim_duration=request_data.get("sim_duration", 12),
                use_hdx_data=request_data.get("use_hdx_data", True),
                use_iati_data=request_data.get("use_iati_data", True),
                hdx_api_key=request_data.get("hdx_api_key")
            )
        else:
            # Convert any arrays in params_to_vary to tuples
            params_to_vary = request_data.get("params_to_vary", {})
            converted_params = {}
            for param, value_range in params_to_vary.items():
                # Convert list/array to tuple if necessary
                if isinstance(value_range, list):
                    converted_params[param] = tuple(value_range)
                else:
                    converted_params[param] = value_range
            
            return analyzer.run_mcp_sensitivity_analysis(
                params_to_vary=converted_params,
                metrics=request_data.get("metrics", ["Insurer Profit"]),
                sim_duration=request_data.get("sim_duration", 12),
                use_hdx_data=request_data.get("use_hdx_data", True),
                use_iati_data=request_data.get("use_iati_data", True),
                hdx_api_key=request_data.get("hdx_api_key"),
                num_points=request_data.get("num_points", 5)
            )
    except Exception as e:
        logging.error(f"API error: {e}")
        return {"error": str(e)}

# Example usage - this would be called by the MCP server
if __name__ == "__main__":
    # Example of a structured parameter analysis request
    example_request = {
        "params_to_vary": {
            "premium_rate": (1.0, 3.0),
            "claim_trigger": (0.3, 0.7)
        },
        "metrics": ["Insurer Profit", "Average Organization Balance"],
        "sim_duration": 12,
        "use_hdx_data": True,
        "use_iati_data": True,
        "num_points": 5
    }
    
    results = mcp_sensitivity_analysis_api(example_request)
    print(json.dumps(results, indent=2))
    
    # Example of a natural language query
    nl_request = {
        "query": "How does changing the premium rate from 1% to 3% affect insurer profit?",
        "sim_duration": 12
    }
    
    nl_results = mcp_sensitivity_analysis_api(nl_request)
    print(json.dumps(nl_results, indent=2))