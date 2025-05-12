"""
Transparent Model Visualizations for Business Continuity Insurance

This module provides visualization components to explain model decisions
and thresholds to users.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Arrow
import matplotlib.ticker as mticker

def create_threshold_gauge(org_name, org_type, current_balance, total_budget, claim_trigger=0.5):
    """
    Create a gauge visualization showing how close an organization is to triggering a claim
    
    Args:
        org_name (str): Organization name
        org_type (str): Organization type
        current_balance (float): Current balance
        total_budget (float): Total budget
        claim_trigger (float, optional): Claim trigger threshold (as fraction of budget). Defaults to 0.5.
        
    Returns:
        plotly.graph_objects.Figure: Gauge visualization
    """
    # Calculate threshold value and percentage
    threshold_value = total_budget * claim_trigger
    current_percentage = (current_balance / total_budget) * 100
    threshold_percentage = claim_trigger * 100
    
    # Define colors and zones
    danger_zone = threshold_percentage
    warning_zone = threshold_percentage + 15
    
    # Create gauge figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_percentage,
        domain={'x': [0, 1], 'y': [0, 0.75]},  # Reduce gauge height to 75% of space
        title={'text': f"{org_name} ({org_type})<br><span style='font-size:0.8em;'>Balance as % of Budget</span>"},
        delta={'reference': threshold_percentage, 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'darkblue'},
            'bar': {'color': 'royalblue'},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, danger_zone], 'color': 'rgba(255, 0, 0, 0.7)'},
                {'range': [danger_zone, warning_zone], 'color': 'rgba(255, 255, 0, 0.5)'},
                {'range': [warning_zone, 100], 'color': 'rgba(0, 255, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': threshold_percentage
            }
        }
    ))
    
    # Add annotations for threshold info
    fig.add_annotation(
        x=0.5, y=0.15,
        text=f"Claim Trigger: {threshold_percentage:.1f}%<br>Current: {current_percentage:.1f}%",
        showarrow=False,
        font=dict(size=12)
    )
    
    # Determine claim eligibility
    if current_percentage < threshold_percentage:
        claim_status = "ELIGIBLE FOR CLAIM"
        claim_color = "red"
    else:
        claim_status = "NOT ELIGIBLE FOR CLAIM"
        claim_color = "green"
    
    # Add claim status at the bottom of the chart, under the gauge
    fig.add_annotation(
        x=0.5, y=0.9,  # Position at bottom of chart (90% down)
        text=claim_status,
        showarrow=False,
        font=dict(size=14, color=claim_color, family="Arial Black"),
        bgcolor="rgba(255, 255, 255, 0.9)",  # Add white background for better contrast
        bordercolor=claim_color,
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout with larger bottom margin to fit the status text
    fig.update_layout(
        height=350,  # Increase height slightly
        margin=dict(l=20, r=20, t=50, b=50),  # More bottom margin
    )
    
    return fig

def create_model_decision_explainer(model_params, simulation_results):
    """
    Create a visual explanation of how the model makes decisions
    
    Args:
        model_params (dict): Parameters used in the model
        simulation_results (dict): Results from the simulation
        
    Returns:
        st: Streamlit components for the explainer
    """
    st.subheader("Model Decision Making Process")
    
    # Create tabs for different aspects of the explainer
    tabs = st.tabs(["Claim Triggers", "Risk Calculation", "Payout Mechanism"])
    
    with tabs[0]:
        st.markdown("### How Claim Triggers Work")
        
        # Extract relevant parameters
        claim_trigger = model_params.get('claim_trigger', 0.5)
        waiting_period = model_params.get('waiting_period', 2)
        
        # Create explanation
        st.markdown(f"""
        The model decides when an organization can make an insurance claim based on two key factors:
        
        1. **Balance Threshold**: An organization becomes eligible for a claim when its balance drops below **{claim_trigger*100:.0f}%** of its total budget.
        2. **Waiting Period**: After making a claim, an organization must wait **{waiting_period} months** before it can make another claim.
        
        This design ensures that:
        - Insurance covers significant financial stress (not minor fluctuations)
        - Organizations can't make excessive claims in quick succession
        - The insurance pool remains sustainable over time
        """)
        
        # Create a visual example
        st.markdown("#### Visual Example of Claim Trigger")
        example_cols = st.columns(2)
        
        with example_cols[0]:
            # Example organization
            example_org = {
                "name": "Example Organization",
                "org_type": "NGO",
                "total_budget": 1000000,
                "balance_history": [800000, 700000, 600000, 500000, 450000]
            }
            
            # Show balance vs trigger
            trigger_value = example_org["total_budget"] * claim_trigger
            steps = list(range(len(example_org["balance_history"])))
            
            # Create threshold chart
            fig = go.Figure()
            
            # Add balance line
            fig.add_trace(go.Scatter(
                x=steps,
                y=example_org["balance_history"],
                mode='lines+markers',
                name='Balance',
                line=dict(color='blue', width=2)
            ))
            
            # Add threshold line
            fig.add_trace(go.Scatter(
                x=steps,
                y=[trigger_value] * len(steps),
                mode='lines',
                name=f'Claim Trigger ({claim_trigger*100:.0f}%)',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Mark claim point
            claim_step = 4  # Example: claim at step 4
            fig.add_trace(go.Scatter(
                x=[claim_step],
                y=[example_org["balance_history"][claim_step]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Claim Triggered',
            ))
            
            # Update layout
            fig.update_layout(
                title="Balance vs. Claim Trigger",
                xaxis_title="Month",
                yaxis_title="Balance ($)",
                legend=dict(x=0, y=1.1, orientation='h'),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with example_cols[1]:
            st.markdown("""
            **What happens when a claim is triggered:**
            
            1. The organization's balance drops below the threshold
            2. The system validates eligibility (no recent claims)
            3. A payout is calculated based on need and maximum coverage
            4. Funds are transferred to the organization
            5. The waiting period begins
            
            This process ensures that claims are:
            - Based on objective financial need
            - Fair and transparent
            - Limited to prevent insurance fund depletion
            """)
            
            # Show waiting period example
            example_claims = [
                {"step": 2, "description": "First claim"},
                {"step": 6, "description": "Second claim"},
            ]
            
            example_org_with_claims = {
                "name": "Example Organization",
                "claims": example_claims,
                "sim_duration": 10
            }
            
            eligibility_fig = create_claim_eligibility_timeline(example_org_with_claims, waiting_period)
            st.plotly_chart(eligibility_fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### How Risk is Calculated")
        
        # Extract relevant parameters
        emergency_prob = model_params.get('emergency_probability', 0.05) * 100
        security_risk = model_params.get('security_risk_factor', 0.2) * 100
        
        # Get risk modifiers
        has_hdx_data = simulation_results.get('hdx_params') is not None
        
        st.markdown(f"""
        The model calculates two main types of risk:
        
        1. **Emergency Risk**: Base probability of {emergency_prob:.1f}% per month
        2. **Security Risk**: Base probability of {security_risk:.1f}% per month
        
        Each organization type has different risk modifiers based on their operational profile:
        """)
        
        # Get modifiers from HDX data or use defaults
        if has_hdx_data:
            hdx_params = simulation_results.get('hdx_params', {})
            security_mods = hdx_params.get('security_parameters', {}).get('security_risk_modifiers', {})
            emergency_mods = hdx_params.get('emergency_parameters', {}).get('emergency_probability_modifiers', {})
        else:
            # Default modifiers
            security_mods = {"NGO": 1.0, "UN Agency": 0.5, "Hybrid": 0.8}
            emergency_mods = {"NGO": 1.2, "UN Agency": 0.7, "Hybrid": 1.0}
        
        # Create risk modifier table
        risk_mod_data = {
            "Organization Type": list(security_mods.keys()),
            "Security Risk Modifier": list(security_mods.values()),
            "Emergency Risk Modifier": list(emergency_mods.values())
        }
        
        risk_df = pd.DataFrame(risk_mod_data)
        risk_df["Effective Security Risk"] = risk_df["Security Risk Modifier"] * security_risk
        risk_df["Effective Emergency Risk"] = risk_df["Emergency Risk Modifier"] * emergency_prob
        
        # Format for display
        display_df = risk_df.copy()
        display_df["Security Risk Modifier"] = display_df["Security Risk Modifier"].apply(lambda x: f"{x:.2f}x")
        display_df["Emergency Risk Modifier"] = display_df["Emergency Risk Modifier"].apply(lambda x: f"{x:.2f}x")
        display_df["Effective Security Risk"] = display_df["Effective Security Risk"].apply(lambda x: f"{x:.1f}%")
        display_df["Effective Emergency Risk"] = display_df["Effective Emergency Risk"].apply(lambda x: f"{x:.1f}%")
        
        st.table(display_df)
        
        # Explain data source for risk factors
        data_source = "ACAPS INFORM Severity Index and ACLED conflict data" if has_hdx_data else "Default simulation values"
        st.markdown(f"**Data Source**: {data_source}")
        
        # Create risk visualization
        st.markdown("#### Monthly Risk Variation")
        
        # Check if we have monthly risk factors
        monthly_factors = None
        if has_hdx_data:
            security_params = simulation_results.get('hdx_params', {}).get('security_parameters', {})
            monthly_factors = security_params.get('monthly_risk_factors', None)
        
        if monthly_factors:
            # Create monthly risk visualization
            months = list(monthly_factors.keys())
            factors = list(monthly_factors.values())
            
            month_fig = px.bar(
                x=months,
                y=factors,
                labels={'x': 'Month', 'y': 'Risk Multiplier'},
                title='Monthly Security Risk Variation',
                color=factors,
                color_continuous_scale='Reds'
            )
            
            month_fig.update_layout(height=300)
            st.plotly_chart(month_fig, use_container_width=True)
            
            st.markdown("""
            The monthly risk factors show how security risk varies throughout the year.
            Higher values indicate increased risk (e.g., due to seasonal patterns in conflict).
            These factors multiply the base security risk probability.
            """)
        else:
            st.info("Monthly risk variation data not available.")
    
    with tabs[2]:
        st.markdown("### How Payouts are Calculated")
        
        # Extract relevant parameters
        premium_rate = model_params.get('premium_rate', 2.0)
        payout_cap = model_params.get('payout_cap_multiple', 3.0)
        
        st.markdown(f"""
        When an organization makes a claim, the payout amount is determined by:
        
        1. **Need-based calculation**: The amount required to bring the organization's balance back up to the claim trigger threshold
        2. **Payout cap**: Maximum payout is limited to **{payout_cap}x** the annual premium
        
        **Premium Calculation**:
        - Annual premium: {premium_rate}% of organization's total budget
        - Monthly premium: Annual premium ÷ 12
        
        **Maximum Possible Payout**:
        - Annual premium × {payout_cap} payout cap
        """)
        
        # Create a visual example of payout calculation
        st.markdown("#### Visual Example of Payout Calculation")
        
        # Example parameters
        example_budget = 1000000
        example_balance = 400000
        annual_premium = example_budget * (premium_rate / 100)
        max_payout = annual_premium * payout_cap
        threshold_value = example_budget * claim_trigger
        
        # Calculate needed amount
        amount_needed = threshold_value - example_balance
        actual_payout = min(amount_needed, max_payout)
        
        # Display calculation
        calc_cols = st.columns(2)
        
        with calc_cols[0]:
            st.markdown("**Organization Example:**")
            st.markdown(f"- Total Budget: ${example_budget:,.0f}")
            st.markdown(f"- Current Balance: ${example_balance:,.0f}")
            st.markdown(f"- Annual Premium: ${annual_premium:,.0f}")
            st.markdown(f"- Claim Trigger Threshold: ${threshold_value:,.0f}")
            
            st.markdown("**Payout Calculation:**")
            st.markdown(f"1. Amount needed: ${amount_needed:,.0f}")
            st.markdown(f"2. Maximum payout: ${max_payout:,.0f}")
            st.markdown(f"3. Actual payout: ${actual_payout:,.0f}")
            
            if amount_needed > max_payout:
                st.markdown("*Payout capped at maximum*")
            else:
                st.markdown("*Full amount needed provided*")
        
        with calc_cols[1]:
            # Create payout visualization
            fig = go.Figure()
            
            # Add current balance bar
            fig.add_trace(go.Bar(
                x=['Current Balance'],
                y=[example_balance],
                name='Current Balance',
                marker_color='red'
            ))
            
            # Add payout bar
            fig.add_trace(go.Bar(
                x=['After Payout'],
                y=[example_balance + actual_payout],
                name='After Payout',
                marker_color='green'
            ))
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=-0.5, y0=threshold_value,
                x1=1.5, y1=threshold_value,
                line=dict(color="blue", width=2, dash="dash"),
            )
            
            # Add annotation for trigger threshold
            fig.add_annotation(
                x=1.5, y=threshold_value,
                text="Claim Trigger Threshold",
                showarrow=True,
                arrowhead=1,
                ax=50,
                ay=-20
            )
            
            # Update layout
            fig.update_layout(
                height=300,
                title="Balance Before and After Payout",
                yaxis_title="Balance ($)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show benefits of this approach
            st.markdown("""
            **Benefits of this payout mechanism:**
            
            1. **Objective**: Based on measurable financial data
            2. **Fair**: Provides exactly what's needed (up to cap)
            3. **Sustainable**: Caps ensure the insurance fund remains viable
            4. **Transparent**: Organizations know exactly how payouts are calculated
            """)
    
    return None

def create_event_impact_diagram(org_data, risk_events, step_range=None):
    """
    Create a diagram showing how risk events impacted organization balance
    
    Args:
        org_data (dict): Organization data including balance history
        risk_events (list): List of risk events from event log
        step_range (tuple): Optional range of steps to show (min_step, max_step)
        
    Returns:
        plotly.graph_objects.Figure: Event impact visualization
    """
    # Extract org name and balance history
    org_name = org_data.get('name', 'Unknown')
    balance_history = org_data.get('balance_history', [])
    
    # Filter events by org and type
    org_events = [e for e in risk_events if org_name in e['description']]
    
    # Determine step range
    if step_range:
        min_step, max_step = step_range
    elif balance_history:
        min_step, max_step = 0, len(balance_history) - 1
    else:
        return None  # No data to visualize
    
    # Filter data by step range
    balance_subset = balance_history[min_step:max_step+1]
    steps = list(range(min_step, max_step+1))
    
    # Create figure
    fig = go.Figure()
    
    # Add balance line
    fig.add_trace(go.Scatter(
        x=steps,
        y=balance_subset,
        mode='lines+markers',
        name=f'{org_name} Balance',
        line=dict(color='blue', width=2)
    ))
    
    # Add events as markers
    for event in org_events:
        step = event.get('step')
        if step is not None and min_step <= step <= max_step:
            event_idx = step - min_step
            if event_idx < len(balance_subset):
                event_balance = balance_subset[event_idx]
                event_type = event.get('type', 'Unknown')
                
                # Determine marker color by event type
                color = 'red' if 'emergency' in event_type.lower() else \
                        'orange' if 'security' in event_type.lower() else \
                        'green' if 'claim' in event_type.lower() else 'gray'
                
                # Add event marker
                fig.add_trace(go.Scatter(
                    x=[step],
                    y=[event_balance],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=color,
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    name=event_type,
                    text=event.get('description'),
                    hoverinfo='text'
                ))
    
    # Update layout
    fig.update_layout(
        title=f'{org_name} Balance with Risk Events',
        xaxis_title='Simulation Step (Month)',
        yaxis_title='Balance ($)',
        hovermode='closest',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def create_acaps_severity_explainer(acaps_data, thresholds=None):
    """
    Create visualization explaining ACAPS severity thresholds and data
    
    Args:
        acaps_data (list): ACAPS data from the API
        thresholds (dict): Threshold values for crisis levels
        
    Returns:
        plotly.graph_objects.Figure: ACAPS severity visualization
    """
    # Use default thresholds if none provided
    if thresholds is None:
        thresholds = {'crisis': 3.5, 'emergency': 4.0, 'catastrophe': 4.5}
    
    # Create DataFrame for visualization
    df = pd.DataFrame(acaps_data)
    
    # Convert severity to numeric
    df['severity'] = pd.to_numeric(df['INFORM Severity Index'], errors='coerce')
    df = df.dropna(subset=['severity'])
    
    # Create histogram of severity scores
    fig = px.histogram(
        df, 
        x='severity',
        nbins=20,
        labels={'severity': 'INFORM Severity Index'},
        title='Distribution of ACAPS INFORM Severity Index Scores',
        color_discrete_sequence=['royalblue']
    )
    
    # Add threshold lines
    for level, value in thresholds.items():
        fig.add_vline(
            x=value, 
            line_width=2, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"{level.title()} ≥ {value}",
            annotation_position="top right" if level == 'crisis' else "top left"
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title='INFORM Severity Index',
        yaxis_title='Count',
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def create_claim_eligibility_timeline(org_data, waiting_period):
    """
    Create a visualization showing when an organization is eligible to make claims
    
    Args:
        org_data (dict): Organization data including claim history
        waiting_period (int): Waiting period between claims in months
        
    Returns:
        plotly.graph_objects.Figure: Claim eligibility timeline
    """
    # Extract org name and claim history
    org_name = org_data.get('name', 'Unknown')
    claims = org_data.get('claims', [])
    sim_duration = org_data.get('sim_duration', 12)
    
    # Create figure
    fig = go.Figure()
    
    # Create base timeline
    steps = list(range(sim_duration))
    
    # Determine eligibility status for each step
    eligibility = []
    for step in steps:
        # Check if any claim's waiting period affects this step
        blocked = False
        for claim in claims:
            claim_step = claim.get('step')
            if claim_step is not None and claim_step <= step and step < claim_step + waiting_period:
                blocked = True
                break
        
        eligibility.append(0 if blocked else 1)  # 0 = Not eligible, 1 = Eligible
    
    # Add eligibility timeline
    fig.add_trace(go.Scatter(
        x=steps,
        y=eligibility,
        mode='lines',
        line=dict(color='green', width=10),
        name='Eligible for Claims'
    ))
    
    # Add claim markers
    for claim in claims:
        step = claim.get('step')
        if step is not None and 0 <= step < sim_duration:
            # Add claim marker
            fig.add_trace(go.Scatter(
                x=[step],
                y=[1],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='circle',
                    line=dict(width=2, color='black')
                ),
                name=f'Claim at Step {step}',
                text=claim.get('description', f'Claim made at step {step}'),
                hoverinfo='text'
            ))
            
            # Add waiting period highlight
            wait_end = min(step + waiting_period, sim_duration)
            fig.add_shape(
                type="rect",
                x0=step,
                y0=-0.1,
                x1=wait_end,
                y1=0.1,
                fillcolor="red",
                opacity=0.3,
                line_width=0,
            )
    
    # Update layout
    fig.update_layout(
        title=f'{org_name} Claim Eligibility Timeline',
        xaxis_title='Simulation Step (Month)',
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Not Eligible', 'Eligible'],
            range=[-0.2, 1.2]
        ),
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig