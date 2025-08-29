"""
Sudan HDX Data Timeline Map Visualization

Interactive map showing Emergency (HAPI/ACLED) and Security (ACAPS) events
with timeline animation where 1 second = 1 month of data.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import time

# Sudan geographic center coordinates
SUDAN_CENTER = {"lat": 15.5007, "lon": 32.5599}
SUDAN_BOUNDS = {
    "north": 22.0, "south": 9.0, 
    "east": 38.5, "west": 21.8
}

def load_integrated_data(use_acled_emergency=False, use_acaps_security=False, use_dtm_emergency=False):
    """Load data from enhanced integrations"""
    try:
        from enhanced_hdx_integration import get_enhanced_hdx_parameters
        
        # Get data from enhanced integration
        integrated_data = get_enhanced_hdx_parameters(
            use_acled_emergency=use_acled_emergency,
            use_acaps_security=use_acaps_security,
            use_dtm_emergency=use_dtm_emergency
        )
        
        if integrated_data:
            emergency_data = integrated_data.get('emergency_parameters')
            security_data = integrated_data.get('security_parameters')
            dtm_data = integrated_data.get('dtm_parameters')
            return emergency_data, security_data, dtm_data
        
        return None, None, None
    except Exception as e:
        st.error(f"Error loading integrated data: {e}")
        return None, None, None

def load_hdx_data():
    """Load and process HDX data from cache files (legacy fallback)"""
    try:
        # Load HAPI/ACLED data (for Emergency events) and ACAPS data (for Security events)
        hapi_file = os.path.join("hdx_cache", "hapi_acled_security_data.json")
        acaps_file = os.path.join("hdx_cache", "acaps_emergency_data.json")
        dtm_file = os.path.join("hdx_cache", "dtm_sudan_displacement.json")
        
        emergency_data = None
        security_data = None
        dtm_data = None
        
        if os.path.exists(hapi_file):
            with open(hapi_file, 'r') as f:
                emergency_data = json.load(f)
        
        if os.path.exists(acaps_file):
            with open(acaps_file, 'r') as f:
                security_data = json.load(f)
        
        if os.path.exists(dtm_file):
            with open(dtm_file, 'r') as f:
                dtm_data = json.load(f)
                
        return emergency_data, security_data, dtm_data
    except Exception as e:
        st.error(f"Error loading HDX data: {e}")
        return None, None, None

def generate_map_events(emergency_data, security_data, dtm_data=None):
    """Generate synthetic geographic events from the HDX data for visualization"""
    events = []
    
    # Generate ACLED Emergency events (new enhanced integration)
    if emergency_data and emergency_data.get('data_source') == 'ACLED API':
        # Handle ACLED data from enhanced integration
        conflict_stats = emergency_data.get('conflict_statistics', {})
        monthly_factors = emergency_data.get('monthly_risk_factors', {})
        
        # Generate events based on conflict statistics
        total_events = max(conflict_stats.get('total_events', 0), 3)  # Show at least 3 for visualization
        emergency_prob = emergency_data.get('emergency_probability', 0.25)
        
        for month_idx, (month, factor) in enumerate(monthly_factors.items(), 1):
            # Create conflict events for visualization
            num_events = max(1, int(total_events * factor / 12))  # Distribute across months
            
            for i in range(num_events):
                events.append({
                    'type': 'Emergency',
                    'event_type': 'Armed Conflict',
                    'month': month,
                    'month_idx': month_idx,
                    'lat': SUDAN_CENTER['lat'] + np.random.uniform(-3, 3),
                    'lon': SUDAN_CENTER['lon'] + np.random.uniform(-4, 4),
                    'severity': emergency_prob,
                    'description': f"Conflict Event: ACLED data shows {emergency_prob:.1%} emergency probability for {month}",
                    'source': 'ACLED Conflict Data',
                    'color': '#FF4444'  # Red for conflict
                })
    
    # Generate Emergency events from HAPI/ACLED data (legacy)
    elif emergency_data and 'event_type_distribution' in emergency_data:
        event_types = emergency_data['event_type_distribution']
        monthly_factors = emergency_data.get('monthly_risk_factors', {})
        
        for month_idx, (month, factor) in enumerate(monthly_factors.items(), 1):
            for event_type, probability in event_types.items():
                # Create events based on probability and monthly factor
                if np.random.random() < (probability * factor * 0.3):  # Scale down for visualization
                    events.append({
                        'type': 'Emergency',
                        'event_type': event_type.replace('_', ' ').title(),
                        'month': month,
                        'month_idx': month_idx,
                        'lat': SUDAN_CENTER['lat'] + np.random.uniform(-3, 3),
                        'lon': SUDAN_CENTER['lon'] + np.random.uniform(-4, 4),
                        'severity': np.random.uniform(0.3, 1.0),
                        'description': f"{event_type.replace('_', ' ').title()} reported in Sudan",
                        'source': 'HAPI/ACLED Conflict Events',
                        'color': '#FF4444'  # Red for emergency
                    })
    
    # Generate Security events from ACAPS data
    if security_data and 'raw_acaps_data' in security_data:
        acaps_crises = security_data['raw_acaps_data']
        
        for month_idx in range(1, 13):
            month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month_idx-1]
            
            # Create security events based on ACAPS severity
            for crisis in acaps_crises[:3]:  # Limit to first few crises for visualization
                severity_index = crisis.get('INFORM Severity Index', 4.0)
                if severity_index >= 4.0:  # High severity events
                    events.append({
                        'type': 'Security',
                        'event_type': crisis.get('crisis_name', 'Security Alert'),
                        'month': month,
                        'month_idx': month_idx,
                        'lat': SUDAN_CENTER['lat'] + np.random.uniform(-2.5, 2.5),
                        'lon': SUDAN_CENTER['lon'] + np.random.uniform(-3.5, 3.5),
                        'severity': severity_index / 5.0,  # Normalize to 0-1
                        'description': f"Security Alert: {crisis.get('crisis_name', 'Unknown')} - Severity: {crisis.get('INFORM Severity category', 'High')}",
                        'source': 'ACAPS INFORM Severity Index',
                        'color': '#FF8800'  # Orange for security
                    })
    
    # Generate DTM Displacement events
    if dtm_data and 'monthly_risk_factors' in dtm_data:
        displacement_stats = dtm_data.get('displacement_statistics', {})
        total_idps = displacement_stats.get('total_idps', 0)
        emergency_prob = dtm_data.get('emergency_probability', 0.18)
        
        # Create displacement events based on IDP numbers OR emergency probability for fallback data
        if total_idps > 0:
            monthly_displacement_intensity = total_idps / 12  # Distribute across months
        else:
            # Use emergency probability to generate events for visualization (fallback mode)
            monthly_displacement_intensity = emergency_prob * 100000  # Scale for visualization
        
        # Generate displacement events for each month
        for month_idx in range(1, 13):
            month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month_idx-1]
            
            # Create displacement events with seasonal variation
            monthly_factor = dtm_data.get('monthly_risk_factors', {}).get(month, 1.0)
            monthly_events = int(monthly_displacement_intensity * monthly_factor / 50000)  # Scale down
            
            for i in range(max(1, monthly_events)):
                events.append({
                        'type': 'Displacement',
                        'event_type': 'IDP Movement',
                        'month': month,
                        'month_idx': month_idx,
                        'lat': SUDAN_CENTER['lat'] + np.random.uniform(-3, 3),
                        'lon': SUDAN_CENTER['lon'] + np.random.uniform(-4, 4),
                        'severity': min(monthly_factor / 2.0, 1.0),  # Normalize severity
                        'description': f"Displacement: {int(monthly_displacement_intensity * monthly_factor):,} IDPs estimated for {month}",
                        'source': 'DTM Displacement Tracking Matrix',
                        'color': '#8800FF'  # Purple for displacement
                    })
    
    return sorted(events, key=lambda x: x['month_idx'])

def create_timeline_map(events):
    """Create interactive timeline map with play controls"""
    
    if not events:
        st.warning("No events data available for visualization")
        return
    
    # Create DataFrame
    df = pd.DataFrame(events)
    
    # Map controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        play_button = st.button("▶️ Play", key="play_map")
    with col2:
        pause_button = st.button("⏸️ Pause", key="pause_map")
    with col3:
        reset_button = st.button("🔄 Reset", key="reset_map")
    with col4:
        month_slider = st.slider("Month", 1, 12, 1, key="month_slider")
    
    # Initialize session state for animation
    if 'map_playing' not in st.session_state:
        st.session_state.map_playing = False
    if 'current_month' not in st.session_state:
        st.session_state.current_month = 1
    
    # Handle controls
    if play_button:
        st.session_state.map_playing = True
    if pause_button:
        st.session_state.map_playing = False
    if reset_button:
        st.session_state.current_month = 1
        st.session_state.map_playing = False
    
    # Use slider value if not playing
    if not st.session_state.map_playing:
        st.session_state.current_month = month_slider
    
    # Filter events up to current month
    current_events = df[df['month_idx'] <= st.session_state.current_month]
    
    if current_events.empty:
        st.info("No events to display for selected time period")
        return
    
    # Create map
    fig = go.Figure()
    
    # Add background map of Sudan
    fig.add_trace(go.Scattermapbox(
        lat=[SUDAN_CENTER['lat']],
        lon=[SUDAN_CENTER['lon']],
        mode='markers',
        marker=dict(size=1, opacity=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add Emergency events (red)
    emergency_events = current_events[current_events['type'] == 'Emergency']
    if not emergency_events.empty:
        fig.add_trace(go.Scattermapbox(
            lat=emergency_events['lat'],
            lon=emergency_events['lon'],
            mode='markers',
            marker=dict(
                size=[s*20 + 10 for s in emergency_events['severity']],
                color='red',
                opacity=0.8
            ),
            text=emergency_events['description'],
            hovertemplate="<b>Emergency Event</b><br>" +
                         "Type: %{text}<br>" +
                         "Month: " + emergency_events['month'].astype(str) + "<br>" +
                         "Source: " + emergency_events['source'] + "<br>" +
                         "<extra></extra>",
            name="Emergency (HAPI/ACLED)",
            showlegend=True
        ))
    
    # Add Security events (orange)
    security_events = current_events[current_events['type'] == 'Security']
    if not security_events.empty:
        fig.add_trace(go.Scattermapbox(
            lat=security_events['lat'],
            lon=security_events['lon'],
            mode='markers',
            marker=dict(
                size=[s*20 + 10 for s in security_events['severity']],
                color='orange',
                opacity=0.8
            ),
            text=security_events['description'],
            hovertemplate="<b>Security Event</b><br>" +
                         "Type: %{text}<br>" +
                         "Month: " + security_events['month'].astype(str) + "<br>" +
                         "Source: " + security_events['source'] + "<br>" +
                         "<extra></extra>",
            name="Security (ACAPS)",
            showlegend=True
        ))
    
    # Add Displacement events (purple)
    displacement_events = current_events[current_events['type'] == 'Displacement']
    if not displacement_events.empty:
        fig.add_trace(go.Scattermapbox(
            lat=displacement_events['lat'],
            lon=displacement_events['lon'],
            mode='markers',
            marker=dict(
                size=[s*15 + 8 for s in displacement_events['severity']],
                color='purple',
                opacity=0.7
            ),
            text=displacement_events['description'],
            hovertemplate="<b>Displacement Event</b><br>" +
                         "Type: %{text}<br>" +
                         "Month: " + displacement_events['month'].astype(str) + "<br>" +
                         "Source: " + displacement_events['source'] + "<br>" +
                         "<extra></extra>",
            name="Displacement (DTM)",
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=SUDAN_CENTER['lat'], lon=SUDAN_CENTER['lon']),
            zoom=6
        ),
        title=f"Sudan HDX Events Timeline - Month {st.session_state.current_month}/12 (2023)",
        height=600,
        showlegend=True
    )
    
    # Display map
    map_container = st.empty()
    map_container.plotly_chart(fig, use_container_width=True)
    
    # Animation logic
    if st.session_state.map_playing and st.session_state.current_month < 12:
        time.sleep(1)  # 1 second = 1 month
        st.session_state.current_month += 1
        st.rerun()
    elif st.session_state.map_playing and st.session_state.current_month >= 12:
        st.session_state.map_playing = False
        st.success("Timeline animation completed!")

def show_event_statistics(events):
    """Display statistics about the events"""
    if not events:
        return
    
    df = pd.DataFrame(events)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Events", len(df))
    
    with col2:
        emergency_count = len(df[df['type'] == 'Emergency'])
        st.metric("Emergency Events", emergency_count)
    
    with col3:
        security_count = len(df[df['type'] == 'Security'])
        st.metric("Security Events", security_count)
    
    # Monthly distribution
    monthly_counts = df.groupby(['month', 'type']).size().unstack(fill_value=0)
    if not monthly_counts.empty:
        st.subheader("Monthly Event Distribution")
        st.bar_chart(monthly_counts)

def main(use_acled_emergency=False, use_acaps_security=False, use_dtm_emergency=False):
    """Main function to run the Sudan timeline map"""
    st.title("🗺️ Sudan HDX Events Timeline Map")
    
    st.markdown("""
    **Interactive visualization of Emergency and Security events in Sudan (2023)**
    
    - **🔴 Emergency Events**: ACLED conflict data (civilian targeting, demonstrations, political violence)
    - **🟠 Security Events**: ACAPS INFORM Severity Index data (complex crises, security alerts)
    - **Timeline**: 1 second = 1 month of data
    """)
    
    # Load data based on user selections
    with st.spinner("Loading HDX data..."):
        # Load data from enhanced integrations when available
        emergency_data, security_data, dtm_data = load_integrated_data(
            use_acled_emergency=use_acled_emergency,
            use_acaps_security=use_acaps_security, 
            use_dtm_emergency=use_dtm_emergency
        )
        
        # Fallback to legacy data if integrated data not available
        if not emergency_data and not security_data and not dtm_data:
            emergency_data, security_data, dtm_data = load_hdx_data()
            
            # Filter data based on user selections for legacy data
            if not use_acled_emergency:
                emergency_data = None
            if not use_acaps_security:
                security_data = None  
            if not use_dtm_emergency:
                dtm_data = None
    
    if not emergency_data and not security_data and not dtm_data:
        st.error("No data available. Please ensure the simulation has been run with HDX or DTM data enabled.")
        return
    
    # Generate events
    with st.spinner("Processing events for visualization..."):
        events = generate_map_events(emergency_data, security_data, dtm_data)
    
    if not events:
        st.warning("No events generated from HDX data.")
        return
    
    # Show statistics
    show_event_statistics(events)
    
    # Create and display map
    st.subheader("Timeline Map")
    create_timeline_map(events)
    
    # Data source info
    with st.expander("📊 Data Sources"):
        st.markdown("""
        **Data Source Assignment (Corrected):**
        - **Emergency Events**: ACLED Conflict Events API + DTM Displacement Data
        - **Security Events**: ACAPS INFORM Severity Index
        
        **Geographic Coverage**: Sudan (SDN)
        **Time Period**: 2023 (12 months)
        **Update Frequency**: 7-day cache refresh
        """)

if __name__ == "__main__":
    main()