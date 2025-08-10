import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import hashlib
import uuid
import warnings
warnings.filterwarnings('ignore')

# Database setup and initialization
def init_database():
    conn = sqlite3.connect('water_management.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id TEXT PRIMARY KEY, name TEXT, zone TEXT, phone TEXT, 
                  last_fetch TIMESTAMP, fetch_count INTEGER DEFAULT 0)''')
    
    # Water usage logs
    c.execute('''CREATE TABLE IF NOT EXISTS usage_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, 
                  timestamp TIMESTAMP, amount REAL, zone TEXT, weather TEXT)''')
    
    # Power source logs
    c.execute('''CREATE TABLE IF NOT EXISTS power_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TIMESTAMP,
                  source TEXT, efficiency REAL, cost REAL)''')
    
    # Water tank status
    c.execute('''CREATE TABLE IF NOT EXISTS tank_status
                 (zone TEXT PRIMARY KEY, current_level REAL, max_capacity REAL,
                  last_refill TIMESTAMP, status TEXT)''')
    
    conn.commit()
    conn.close()

# Generate sample data for demonstration
def generate_sample_data():
    conn = sqlite3.connect('water_management.db')
    
    # Check if data already exists
    existing_users = pd.read_sql("SELECT COUNT(*) as count FROM users", conn)
    if existing_users['count'].iloc[0] > 0:
        conn.close()
        return
    
    # Sample zones in Malete
    zones = ['Zone A - Central', 'Zone B - North', 'Zone C - South', 'Zone D - East', 'Zone E - West']
    
    # Generate sample users
    users_data = []
    for i in range(100):
        user_id = f"MLT{str(i+1000).zfill(4)}"
        zone = np.random.choice(zones)
        last_fetch = datetime.now() - timedelta(hours=np.random.randint(1, 72))
        users_data.append((user_id, f"User {i+1}", zone, f"080{np.random.randint(10000000, 99999999)}", 
                          last_fetch, np.random.randint(0, 3)))
    
    users_df = pd.DataFrame(users_data, columns=['id', 'name', 'zone', 'phone', 'last_fetch', 'fetch_count'])
    users_df.to_sql('users', conn, if_exists='replace', index=False)
    
    # Generate historical usage data
    usage_data = []
    weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Hot']
    for i in range(1000):
        user_id = f"MLT{str(np.random.randint(1000, 1099)).zfill(4)}"
        timestamp = datetime.now() - timedelta(days=np.random.randint(1, 365))
        amount = np.random.normal(20, 5)  # liters
        zone = np.random.choice(zones)
        weather = np.random.choice(weather_conditions)
        usage_data.append((user_id, timestamp, max(5, amount), zone, weather))
    
    usage_df = pd.DataFrame(usage_data, columns=['user_id', 'timestamp', 'amount', 'zone', 'weather'])
    usage_df.to_sql('usage_logs', conn, if_exists='replace', index=False)
    
    # Generate power source data
    power_data = []
    power_sources = ['Solar', 'Generator', 'Grid']
    for i in range(200):
        timestamp = datetime.now() - timedelta(hours=np.random.randint(1, 720))
        source = np.random.choice(power_sources)
        if source == 'Solar':
            efficiency = np.random.uniform(0.7, 0.95)
            cost = 0
        elif source == 'Generator':
            efficiency = np.random.uniform(0.6, 0.8)
            cost = np.random.uniform(500, 1500)
        else:  # Grid
            efficiency = np.random.uniform(0.8, 0.9)
            cost = np.random.uniform(200, 800)
        power_data.append((timestamp, source, efficiency, cost))
    
    power_df = pd.DataFrame(power_data, columns=['timestamp', 'source', 'efficiency', 'cost'])
    power_df.to_sql('power_logs', conn, if_exists='replace', index=False)
    
    # Initialize tank status
    tank_data = []
    for zone in zones:
        current_level = np.random.uniform(500, 1000)
        max_capacity = 1000
        last_refill = datetime.now() - timedelta(hours=np.random.randint(1, 24))
        status = 'Normal' if current_level > 300 else 'Low'
        tank_data.append((zone, current_level, max_capacity, last_refill, status))
    
    tank_df = pd.DataFrame(tank_data, columns=['zone', 'current_level', 'max_capacity', 'last_refill', 'status'])
    tank_df.to_sql('tank_status', conn, if_exists='replace', index=False)
    
    conn.close()

# XGBoost model for water usage prediction
class WaterUsagePredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = ['hour', 'day_of_week', 'month', 'zone_encoded', 'weather_encoded', 'historical_avg']
    
    def prepare_features(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Encode categorical variables
        zone_mapping = {zone: i for i, zone in enumerate(df['zone'].unique())}
        weather_mapping = {weather: i for i, weather in enumerate(df['weather'].unique())}
        
        df['zone_encoded'] = df['zone'].map(zone_mapping)
        df['weather_encoded'] = df['weather'].map(weather_mapping)
        
        # Calculate historical average
        df['historical_avg'] = df.groupby('zone')['amount'].transform('mean')
        
        return df[self.feature_columns], df['amount']
    
    def train(self, df):
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return mae, r2
    
    def predict(self, features_df):
        if self.model is None:
            return np.array([20.0] * len(features_df))  # Default prediction
        
        X, _ = self.prepare_features(features_df)
        return self.model.predict(X)

# Initialize database and generate sample data
init_database()
generate_sample_data()

# Streamlit App Configuration
st.set_page_config(
    page_title="Malete Water Management System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
    }
    .alert-normal {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🌊 Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "Dashboard", "User Management", "Water Predictions", "Power Management", "Admin Panel"
])

# Main Title
st.markdown('<h1 class="main-header">💧 Malete Smart Water Management System</h1>', unsafe_allow_html=True)

# Dashboard Page
if page == "Dashboard":
    st.header("📊 Real-time Dashboard")
    
    # Connect to database
    conn = sqlite3.connect('water_management.db')
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    # Total active users
    total_users = pd.read_sql("SELECT COUNT(*) as count FROM users", conn)['count'].iloc[0]
    with col1:
        st.metric("👥 Active Users", total_users)
    
    # Total water distributed today
    today = datetime.now().date()
    today_usage = pd.read_sql(f"""
        SELECT COALESCE(SUM(amount), 0) as total 
        FROM usage_logs 
        WHERE DATE(timestamp) = '{today}'
    """, conn)['total'].iloc[0]
    with col2:
        st.metric("💧 Water Today (L)", f"{today_usage:.1f}")
    
    # Active zones
    tank_status = pd.read_sql("SELECT * FROM tank_status", conn)
    active_zones = len(tank_status[tank_status['status'] == 'Normal'])
    with col3:
        st.metric("🏘️ Active Zones", f"{active_zones}/5")
    
    # Current power source
    latest_power = pd.read_sql("""
        SELECT source FROM power_logs 
        ORDER BY timestamp DESC LIMIT 1
    """, conn)
    current_power = latest_power['source'].iloc[0] if not latest_power.empty else "Unknown"
    with col4:
        st.metric("⚡ Power Source", current_power)
    
    st.divider()
    
    # Water Tank Status
    st.subheader("🏘️ Zone Water Tank Status")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tank levels chart
        fig = go.Figure()
        for _, tank in tank_status.iterrows():
            color = 'green' if tank['current_level'] > 300 else 'orange' if tank['current_level'] > 150 else 'red'
            fig.add_trace(go.Bar(
                x=[tank['zone']],
                y=[tank['current_level']],
                name=tank['zone'],
                marker_color=color,
                showlegend=False
            ))
        
        fig.add_hline(y=300, line_dash="dash", line_color="red", 
                     annotation_text="Critical Level (300L)")
        fig.update_layout(
            title="Water Tank Levels by Zone",
            xaxis_title="Zones",
            yaxis_title="Water Level (Liters)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Zone Status")
        for _, tank in tank_status.iterrows():
            if tank['current_level'] <= 150:
                st.markdown(f'<div class="alert-high"><strong>{tank["zone"]}</strong><br>Level: {tank["current_level"]:.0f}L<br>Status: Critical</div>', unsafe_allow_html=True)
            elif tank['current_level'] <= 300:
                st.markdown(f'<div class="alert-medium"><strong>{tank["zone"]}</strong><br>Level: {tank["current_level"]:.0f}L<br>Status: Low</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-normal"><strong>{tank["zone"]}</strong><br>Level: {tank["current_level"]:.0f}L<br>Status: Normal</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Usage Trends
    st.subheader("📈 Water Usage Trends")
    
    # Get recent usage data
    recent_usage = pd.read_sql("""
        SELECT DATE(timestamp) as date, SUM(amount) as total_usage, zone
        FROM usage_logs 
        WHERE timestamp >= datetime('now', '-30 days')
        GROUP BY DATE(timestamp), zone
        ORDER BY date
    """, conn)
    
    if not recent_usage.empty:
        recent_usage['date'] = pd.to_datetime(recent_usage['date'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily usage trend
            daily_total = recent_usage.groupby('date')['total_usage'].sum().reset_index()
            fig_trend = px.line(daily_total, x='date', y='total_usage',
                               title='Daily Water Usage Trend (Last 30 Days)',
                               labels={'total_usage': 'Usage (Liters)', 'date': 'Date'})
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Usage by zone
            zone_usage = recent_usage.groupby('zone')['total_usage'].sum().reset_index()
            fig_zone = px.pie(zone_usage, values='total_usage', names='zone',
                             title='Water Usage Distribution by Zone')
            st.plotly_chart(fig_zone, use_container_width=True)
    
    conn.close()

# User Management Page
elif page == "User Management":
    st.header("👥 User Management")
    
    conn = sqlite3.connect('water_management.db')
    
    # User registration form
    st.subheader("📝 Register New User")
    with st.form("user_registration"):
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("Full Name")
            new_phone = st.text_input("Phone Number")
        with col2:
            zones = ['Zone A - Central', 'Zone B - North', 'Zone C - South', 'Zone D - East', 'Zone E - West']
            new_zone = st.selectbox("Zone", zones)
        
        if st.form_submit_button("Register User"):
            if new_name and new_phone:
                # Generate unique user ID
                user_id = f"MLT{str(np.random.randint(1100, 9999)).zfill(4)}"
                
                # Insert new user
                c = conn.cursor()
                c.execute("""
                    INSERT INTO users (id, name, zone, phone, last_fetch, fetch_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, new_name, new_zone, new_phone, datetime.now(), 0))
                conn.commit()
                
                st.success(f"✅ User registered successfully! User ID: {user_id}")
            else:
                st.error("Please fill all required fields.")
    
    st.divider()
    
    # Water fetching eligibility checker
    st.subheader("🔍 Check Water Fetching Eligibility")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        user_id_check = st.text_input("Enter User ID")
        if st.button("Check Eligibility"):
            if user_id_check:
                user = pd.read_sql(f"SELECT * FROM users WHERE id = '{user_id_check}'", conn)
                if not user.empty:
                    user_data = user.iloc[0]
                    last_fetch = pd.to_datetime(user_data['last_fetch'])
                    hours_since_fetch = (datetime.now() - last_fetch).total_seconds() / 3600
                    
                    # Check if user can fetch (24-hour cooldown)
                    can_fetch = hours_since_fetch >= 24
                    
                    st.session_state['checked_user'] = {
                        'user_data': user_data,
                        'can_fetch': can_fetch,
                        'hours_since_fetch': hours_since_fetch
                    }
                else:
                    st.error("User ID not found!")
    
    with col2:
        if 'checked_user' in st.session_state:
            user_info = st.session_state['checked_user']
            user_data = user_info['user_data']
            
            st.write("**User Information:**")
            st.write(f"Name: {user_data['name']}")
            st.write(f"Zone: {user_data['zone']}")
            st.write(f"Phone: {user_data['phone']}")
            st.write(f"Last Fetch: {user_data['last_fetch']}")
            st.write(f"Fetch Count: {user_data['fetch_count']}")
            
            if user_info['can_fetch']:
                st.success("✅ User is eligible to fetch water!")
                
                # Water fetching form
                if st.button("Record Water Fetch"):
                    # Update user's last fetch time
                    c = conn.cursor()
                    c.execute("""
                        UPDATE users 
                        SET last_fetch = ?, fetch_count = fetch_count + 1
                        WHERE id = ?
                    """, (datetime.now(), user_data['id']))
                    
                    # Log the usage
                    c.execute("""
                        INSERT INTO usage_logs (user_id, timestamp, amount, zone, weather)
                        VALUES (?, ?, ?, ?, ?)
                    """, (user_data['id'], datetime.now(), 20.0, user_data['zone'], 'Sunny'))
                    
                    conn.commit()
                    st.success("Water fetch recorded successfully!")
                    
                    # Clear the session state to refresh data
                    del st.session_state['checked_user']
                    st.rerun()
            else:
                remaining_hours = 24 - user_info['hours_since_fetch']
                st.warning(f"⏳ User must wait {remaining_hours:.1f} more hours before next fetch.")
    
    st.divider()
    
    # Recent users list
    st.subheader("👥 Recent Users")
    users = pd.read_sql("SELECT * FROM users ORDER BY last_fetch DESC LIMIT 10", conn)
    st.dataframe(users, use_container_width=True)
    
    conn.close()

# Water Predictions Page
elif page == "Water Predictions":
    st.header("🔮 Water Usage Predictions")
    
    conn = sqlite3.connect('water_management.db')
    
    # Load historical data and train model
    usage_data = pd.read_sql("SELECT * FROM usage_logs", conn)
    
    if not usage_data.empty:
        # Initialize and train predictor
        predictor = WaterUsagePredictor()
        
        with st.spinner("Training XGBoost model..."):
            mae, r2 = predictor.train(usage_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model MAE", f"{mae:.2f}L")
        with col2:
            st.metric("Model R² Score", f"{r2:.3f}")
        
        st.divider()
        
        # Prediction interface
        st.subheader("📊 Make Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_zone = st.selectbox("Select Zone", 
                                   ['Zone A - Central', 'Zone B - North', 'Zone C - South', 'Zone D - East', 'Zone E - West'])
            pred_weather = st.selectbox("Weather Condition", ['Sunny', 'Cloudy', 'Rainy', 'Hot'])
        
        with col2:
            pred_date = st.date_input("Prediction Date", datetime.now().date())
            pred_hour = st.slider("Hour of Day", 0, 23, 12)
        
        if st.button("Generate Prediction"):
            # Create prediction data
            pred_data = pd.DataFrame({
                'timestamp': [datetime.combine(pred_date, datetime.min.time().replace(hour=pred_hour))],
                'zone': [pred_zone],
                'weather': [pred_weather],
                'amount': [0]  # Placeholder
            })
            
            prediction = predictor.predict(pred_data)[0]
            
            st.success(f"🔮 Predicted water usage: {prediction:.1f} liters")
        
        st.divider()
        
        # Hourly predictions for next 24 hours
        st.subheader("📈 24-Hour Forecast")
        
        forecast_zone = st.selectbox("Zone for Forecast", 
                                   ['Zone A - Central', 'Zone B - North', 'Zone C - South', 'Zone D - East', 'Zone E - West'],
                                   key="forecast_zone")
        
        if st.button("Generate 24-Hour Forecast"):
            # Generate hourly predictions
            forecast_data = []
            base_time = datetime.now()
            
            for i in range(24):
                pred_time = base_time + timedelta(hours=i)
                pred_data = pd.DataFrame({
                    'timestamp': [pred_time],
                    'zone': [forecast_zone],
                    'weather': ['Sunny'],  # Default weather
                    'amount': [0]
                })
                
                prediction = predictor.predict(pred_data)[0]
                forecast_data.append({
                    'hour': pred_time.strftime('%H:00'),
                    'predicted_usage': prediction
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            
            # Plot forecast
            fig = px.line(forecast_df, x='hour', y='predicted_usage',
                         title=f'24-Hour Water Usage Forecast - {forecast_zone}',
                         labels={'predicted_usage': 'Predicted Usage (L)', 'hour': 'Hour'})
            fig.update_traces(line_color='#1E88E5', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display recommendations
            peak_hour = forecast_df.loc[forecast_df['predicted_usage'].idxmax(), 'hour']
            low_hour = forecast_df.loc[forecast_df['predicted_usage'].idxmin(), 'hour']
            
            st.info(f"""
            📋 **Recommendations:**
            - **Peak Usage Time:** {peak_hour} - Ensure adequate water supply
            - **Optimal Fetch Time:** {low_hour} - Least congestion expected
            - **Average Predicted Usage:** {forecast_df['predicted_usage'].mean():.1f}L per hour
            """)
    
    else:
        st.warning("No historical data available for predictions.")
    
    conn.close()

# Power Management Page
# Power Management Page - Corrected Version
elif page == "Power Management":
    st.header("⚡ Power Source Management")
    
    conn = sqlite3.connect('water_management.db')
    
    # Current power status
    latest_power = pd.read_sql("""
        SELECT * FROM power_logs 
        ORDER BY timestamp DESC LIMIT 1
    """, conn)
    
    if not latest_power.empty:
        current_power = latest_power.iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Add power source emoji/icon
            source_icons = {'Solar': '☀️', 'Generator': '⚡', 'Grid': '🔌'}
            icon = source_icons.get(current_power['source'], '⚡')
            st.metric(f"{icon} Current Source", current_power['source'])
        
        with col2:
            st.metric("⚡ Efficiency", f"{current_power['efficiency']:.1%}")
        
        with col3:
            cost_display = "₦0 (Free)" if current_power['cost'] == 0 else f"₦{current_power['cost']:.0f}/hr"
            st.metric("💰 Operating Cost", cost_display)
        
        # Power source health indicator
        if current_power['efficiency'] < 0.6:
            st.error("🚨 Low efficiency detected! Consider maintenance or switching power source.")
        elif current_power['efficiency'] < 0.8:
            st.warning("⚠️ Moderate efficiency. Monitor performance closely.")
        else:
            st.success("✅ Power source operating at optimal efficiency.")
    else:
        st.warning("⚠️ No power source data available. Please initialize a power source.")
    
    st.divider()
    
    # Power source switching with better validation
    st.subheader("🔄 Switch Power Source")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_source = st.selectbox("Select Power Source", ['Solar', 'Generator', 'Grid'])
        
        # Set realistic default values based on source
        if new_source == 'Solar':
            default_efficiency = 0.85
            default_cost = 0.0
            efficiency_range = (0.60, 0.95)
        elif new_source == 'Generator':
            default_efficiency = 0.70
            default_cost = 800.0
            efficiency_range = (0.50, 0.80)
        else:  # Grid
            default_efficiency = 0.85
            default_cost = 400.0
            efficiency_range = (0.70, 0.90)
        
        # Efficiency slider with appropriate ranges
        efficiency = st.slider(
            f"Efficiency (%) - Range for {new_source}: {efficiency_range[0]:.0%}-{efficiency_range[1]:.0%}", 
            efficiency_range[0], efficiency_range[1], default_efficiency, 0.01,
            help=f"Typical efficiency range for {new_source} power systems"
        )
        
        # Cost input with validation
        if new_source == 'Solar':
            cost = 0.0
            st.info("☀️ Solar power has zero operational cost")
        else:
            max_cost = 2000.0 if new_source == 'Generator' else 1000.0
            cost = st.number_input(
                f"Cost per hour (₦) - Max for {new_source}: ₦{max_cost:.0f}", 
                0.0, max_cost, default_cost,
                help=f"Operational cost per hour for {new_source}"
            )
        
        # Add reason for switch
        switch_reason = st.text_area("Reason for switching (optional)", 
                                   placeholder="e.g., scheduled maintenance, cost optimization, power outage...")
    
    with col2:
        st.write("**Power Source Characteristics:**")
        
        if new_source == 'Solar':
            st.info("""
            🌞 **Solar Power**
            - ✅ Clean & renewable energy
            - ✅ Zero operational cost
            - ⚠️ Weather dependent
            - ⚠️ Daylight hours only
            - 💡 Best efficiency: 10 AM - 3 PM
            """)
        elif new_source == 'Generator':
            st.warning("""
            ⛽ **Generator Power**
            - ✅ Reliable backup option
            - ✅ Weather independent
            - ❌ Higher operational cost
            - ❌ Fuel dependent
            - 🔧 Requires regular maintenance
            """)
        else:
            st.info("""
            🔌 **Grid Electricity**
            - ✅ Stable supply
            - ✅ Moderate cost
            - ⚠️ Grid dependent
            - ⚠️ Outage susceptible
            - 💡 Most reliable option
            """)
        
        # Show current vs new comparison
        if not latest_power.empty:
            current = latest_power.iloc[0]
            st.write("**Comparison with Current:**")
            
            eff_change = efficiency - current['efficiency']
            cost_change = cost - current['cost']
            
            if eff_change > 0:
                st.write(f"📈 Efficiency: +{eff_change:.1%} improvement")
            elif eff_change < 0:
                st.write(f"📉 Efficiency: {eff_change:.1%} decrease")
            else:
                st.write("➡️ Efficiency: No change")
            
            if cost_change > 0:
                st.write(f"📈 Cost: +₦{cost_change:.0f}/hr increase")
            elif cost_change < 0:
                st.write(f"📉 Cost: -₦{abs(cost_change):.0f}/hr savings")
            else:
                st.write("➡️ Cost: No change")
    
    # Confirmation dialog for switching
    if st.button("🔄 Switch Power Source", type="primary"):
        # Add confirmation for critical switches
        should_switch = True
        
        if not latest_power.empty:
            current = latest_power.iloc[0]
            # Warn if switching to less efficient or more expensive option
            if efficiency < current['efficiency'] - 0.1:
                st.warning("⚠️ Warning: New source has significantly lower efficiency!")
            if cost > current['cost'] + 200:
                st.warning("⚠️ Warning: New source has significantly higher cost!")
        
        if should_switch:
            try:
                # Log the power source change with additional info
                c = conn.cursor()
                c.execute("""
                    INSERT INTO power_logs (timestamp, source, efficiency, cost)
                    VALUES (?, ?, ?, ?)
                """, (datetime.now(), new_source, efficiency, cost))
                conn.commit()
                
                # Log the switch reason if provided
                if switch_reason.strip():
                    st.info(f"📝 Switch reason: {switch_reason}")
                
                st.success(f"✅ Power source successfully switched to {new_source}")
                st.success(f"🔧 New configuration: {efficiency:.1%} efficiency, ₦{cost:.0f}/hr cost")
                
                # Auto-refresh to show new data
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error switching power source: {str(e)}")
    
    st.divider()
    
    # Power usage analytics with improved visualizations
    st.subheader("📊 Power Usage Analytics")
    
    power_data = pd.read_sql("""
        SELECT * FROM power_logs 
        ORDER BY timestamp DESC LIMIT 168  -- Last week of hourly data
    """, conn)
    
    if not power_data.empty and len(power_data) > 1:
        power_data['timestamp'] = pd.to_datetime(power_data['timestamp'])
        power_data = power_data.sort_values('timestamp')
        
        # Analytics time period selector
        col1, col2 = st.columns([3, 1])
        
        with col2:
            analysis_period = st.selectbox("Analysis Period", 
                                         ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"])
        
        # Filter data based on selected period
        now = datetime.now()
        if analysis_period == "Last 24 Hours":
            filtered_data = power_data[power_data['timestamp'] >= now - timedelta(hours=24)]
        elif analysis_period == "Last 7 Days":
            filtered_data = power_data[power_data['timestamp'] >= now - timedelta(days=7)]
        elif analysis_period == "Last 30 Days":
            filtered_data = power_data[power_data['timestamp'] >= now - timedelta(days=30)]
        else:
            filtered_data = power_data
        
        if not filtered_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Power source distribution with better colors
                source_dist = filtered_data['source'].value_counts()
                colors = ['#FFD700', '#FF6B6B', '#4ECDC4']  # Gold, Red, Teal
                
                fig_pie = px.pie(
                    values=source_dist.values, 
                    names=source_dist.index,
                    title=f'Power Source Usage - {analysis_period}',
                    color_discrete_sequence=colors
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Efficiency over time
                fig_efficiency = px.line(
                    filtered_data, 
                    x='timestamp', 
                    y='efficiency',
                    color='source',
                    title=f'Power Efficiency Over Time - {analysis_period}',
                    labels={'efficiency': 'Efficiency (%)', 'timestamp': 'Time'}
                )
                fig_efficiency.update_layout(yaxis_tickformat='.0%')
                st.plotly_chart(fig_efficiency, use_container_width=True)
            
            # Cost analysis with cumulative view
            st.subheader("💰 Cost Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily cost breakdown
                if len(filtered_data) > 1:
                    daily_cost = filtered_data.groupby([
                        filtered_data['timestamp'].dt.date, 'source'
                    ])['cost'].sum().reset_index()
                    daily_cost.columns = ['date', 'source', 'total_cost']
                    
                    fig_cost = px.bar(
                        daily_cost, 
                        x='date', 
                        y='total_cost', 
                        color='source',
                        title=f'Daily Power Costs by Source - {analysis_period}',
                        labels={'total_cost': 'Cost (₦)', 'date': 'Date'}
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
            
            with col2:
                # Cost summary metrics
                total_cost = filtered_data['cost'].sum()
                avg_hourly_cost = filtered_data['cost'].mean()
                hours_analyzed = len(filtered_data)
                
                st.metric("💰 Total Cost", f"₦{total_cost:,.0f}")
                st.metric("⏱️ Avg Hourly Cost", f"₦{avg_hourly_cost:.0f}")
                st.metric("🕐 Hours Analyzed", f"{hours_analyzed}")
                
                # Cost breakdown by source
                cost_by_source = filtered_data.groupby('source')['cost'].sum()
                st.write("**Cost by Source:**")
                for source, cost in cost_by_source.items():
                    percentage = (cost / total_cost * 100) if total_cost > 0 else 0
                    st.write(f"• {source}: ₦{cost:,.0f} ({percentage:.1f}%)")
        
        # Power efficiency comparison with statistical analysis
        st.subheader("⚡ Efficiency Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average efficiency by source
            avg_efficiency = filtered_data.groupby('source').agg({
                'efficiency': ['mean', 'std', 'min', 'max', 'count']
            }).round(3)
            avg_efficiency.columns = ['Mean', 'Std Dev', 'Min', 'Max', 'Count']
            
            st.write("**Efficiency Statistics by Source:**")
            st.dataframe(avg_efficiency, use_container_width=True)
        
        with col2:
            # Efficiency comparison chart
            efficiency_stats = filtered_data.groupby('source')['efficiency'].mean().reset_index()
            
            fig_eff = px.bar(
                efficiency_stats, 
                x='source', 
                y='efficiency',
                title='Average Efficiency by Power Source',
                labels={'efficiency': 'Average Efficiency', 'source': 'Power Source'},
                color='source',
                color_discrete_sequence=['#FFD700', '#FF6B6B', '#4ECDC4']
            )
            fig_eff.update_layout(yaxis_tickformat='.0%', showlegend=False)
            fig_eff.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            st.plotly_chart(fig_eff, use_container_width=True)
        
        # Power source recommendations
        st.subheader("💡 Recommendations")
        
        # Calculate recommendations based on data
        most_efficient = avg_efficiency['Mean'].idxmax()
        most_cost_effective = cost_by_source.idxmin() if len(cost_by_source) > 0 else "Solar"
        most_used = source_dist.index[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **🏆 Most Efficient**
            {most_efficient}
            Avg: {avg_efficiency.loc[most_efficient, 'Mean']:.1%}
            """)
        
        with col2:
            st.success(f"""
            **💰 Most Cost-Effective**
            {most_cost_effective}
            Total Cost: ₦{cost_by_source.get(most_cost_effective, 0):,.0f}
            """)
        
        with col3:
            st.warning(f"""
            **📈 Most Used**
            {most_used}
            Usage: {source_dist.iloc[0]} times ({source_dist.iloc[0]/len(filtered_data)*100:.1f}%)
            """)
        
        # Optimization suggestions
        st.write("**🔧 Optimization Suggestions:**")
        
        suggestions = []
        
        # Efficiency-based suggestions
        if most_efficient != most_used and avg_efficiency.loc[most_efficient, 'Mean'] > avg_efficiency.loc[most_used, 'Mean'] + 0.1:
            suggestions.append(f"Consider using {most_efficient} more often for better efficiency")
        
        # Cost-based suggestions
        if most_cost_effective != most_used and most_cost_effective in cost_by_source:
            potential_savings = cost_by_source.get(most_used, 0) - cost_by_source.get(most_cost_effective, 0)
            if potential_savings > 100:
                suggestions.append(f"Switch to {most_cost_effective} to potentially save ₦{potential_savings:,.0f}")
        
        # Maintenance suggestions
        for source in avg_efficiency.index:
            if avg_efficiency.loc[source, 'Mean'] < 0.7:
                suggestions.append(f"Consider maintenance for {source} (efficiency below 70%)")
        
        if suggestions:
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
        else:
            st.write("• Current power management appears optimal!")
    
    else:
        st.info("📊 Insufficient data for detailed analytics. More data will be available after power source switches.")
    
    # Add power monitoring alerts
    st.subheader("🚨 Power Monitoring Alerts")
    
    if not latest_power.empty:
        current = latest_power.iloc[0]
        
        # Check for alerts
        alerts = []
        
        if current['efficiency'] < 0.6:
            alerts.append(("🔴 Critical", f"Efficiency critically low: {current['efficiency']:.1%}"))
        elif current['efficiency'] < 0.75:
            alerts.append(("🟡 Warning", f"Efficiency below optimal: {current['efficiency']:.1%}"))
        
        if current['cost'] > 1500:
            alerts.append(("🔴 Critical", f"High operational cost: ₦{current['cost']:.0f}/hr"))
        elif current['cost'] > 1000:
            alerts.append(("🟡 Warning", f"Elevated cost: ₦{current['cost']:.0f}/hr"))
        
        # Time-based alerts
        last_update = pd.to_datetime(current['timestamp'])
        hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
        
        if hours_since_update > 24:
            alerts.append(("🟡 Warning", f"No power updates for {hours_since_update:.1f} hours"))
        
        if alerts:
            for alert_type, message in alerts:
                if "Critical" in alert_type:
                    st.error(f"{alert_type}: {message}")
                else:
                    st.warning(f"{alert_type}: {message}")
        else:
            st.success("✅ All power systems operating normally")
    
    conn.close()

# Admin Panel Page
elif page == "Admin Panel":
    st.header("🔧 Admin Control Panel")
    
    # Admin authentication (simplified)
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.subheader("🔐 Admin Login")
        admin_password = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if admin_password == "admin123":  # Simple password for demo
                st.session_state.admin_authenticated = True
                st.success("✅ Admin login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid password!")
    else:
        conn = sqlite3.connect('water_management.db')
        
        # Tank level management
        st.subheader("🏘️ Tank Level Management")
        
        tank_status = pd.read_sql("SELECT * FROM tank_status", conn)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Tank Levels:**")
            st.dataframe(tank_status[['zone', 'current_level', 'max_capacity', 'status']], 
                        use_container_width=True)
        
        with col2:
            st.write("**Update Tank Level:**")
            zone_to_update = st.selectbox("Select Zone", tank_status['zone'].tolist())
            new_level = st.number_input("New Water Level (L)", 0.0, 1000.0, 500.0)
            
            if st.button("Update Tank Level"):
                # Update tank level
                c = conn.cursor()
                status = 'Normal' if new_level > 300 else 'Low' if new_level > 150 else 'Critical'
                c.execute("""
                    UPDATE tank_status 
                    SET current_level = ?, last_refill = ?, status = ?
                    WHERE zone = ?
                """, (new_level, datetime.now(), status, zone_to_update))
                conn.commit()
                
                st.success(f"✅ Tank level updated for {zone_to_update}")
                st.rerun()
        
        st.divider()
        
        # System overrides
        st.subheader("⚙️ System Overrides")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Emergency Actions:**")
            
            if st.button("🚨 Emergency Refill All Tanks"):
                c = conn.cursor()
                c.execute("""
                    UPDATE tank_status 
                    SET current_level = max_capacity, 
                        last_refill = ?, 
                        status = 'Normal'
                """, (datetime.now(),))
                conn.commit()
                st.success("✅ All tanks refilled to maximum capacity!")
                st.rerun()
            
            if st.button("🔄 Reset All User Fetch Timers"):
                c = conn.cursor()
                c.execute("""
                    UPDATE users 
                    SET last_fetch = ?
                """, (datetime.now() - timedelta(days=1),))
                conn.commit()
                st.success("✅ All user fetch timers reset!")
                st.rerun()
        
        with col2:
            st.write("**User Management:**")
            
            # User restriction override
            user_id_override = st.text_input("User ID to Override")
            if st.button("Override Fetch Restriction"):
                if user_id_override:
                    c = conn.cursor()
                    c.execute("""
                        UPDATE users 
                        SET last_fetch = ?
                        WHERE id = ?
                    """, (datetime.now() - timedelta(days=1), user_id_override))
                    conn.commit()
                    
                    if c.rowcount > 0:
                        st.success(f"✅ Fetch restriction overridden for {user_id_override}")
                    else:
                        st.error("❌ User ID not found!")
        
        st.divider()
        
        # System statistics
        st.subheader("📈 System Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Total users
        total_users = pd.read_sql("SELECT COUNT(*) as count FROM users", conn)['count'].iloc[0]
        with col1:
            st.metric("Total Users", total_users)
        
        # Total water distributed
        total_water = pd.read_sql("SELECT COALESCE(SUM(amount), 0) as total FROM usage_logs", conn)['total'].iloc[0]
        with col2:
            st.metric("Total Water Distributed", f"{total_water:.0f}L")
        
        # Average daily usage
        avg_daily = pd.read_sql("""
            SELECT AVG(daily_usage) as avg_usage FROM (
                SELECT DATE(timestamp) as date, SUM(amount) as daily_usage
                FROM usage_logs 
                GROUP BY DATE(timestamp)
            )
        """, conn)['avg_usage'].iloc[0] or 0
        with col3:
            st.metric("Avg Daily Usage", f"{avg_daily:.0f}L")
        
        # System uptime (simulated)
        uptime_days = (datetime.now() - datetime(2024, 1, 1)).days
        with col4:
            st.metric("System Uptime", f"{uptime_days} days")
        
        # Recent activity log
        st.subheader("📋 Recent Activity Log")
        
        recent_activity = pd.read_sql("""
            SELECT user_id, timestamp, amount, zone 
            FROM usage_logs 
            ORDER BY timestamp DESC 
            LIMIT 20
        """, conn)
        
        if not recent_activity.empty:
            st.dataframe(recent_activity, use_container_width=True)
        
        # Data export functionality
        st.subheader("💾 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Usage Data"):
                usage_data = pd.read_sql("SELECT * FROM usage_logs", conn)
                csv = usage_data.to_csv(index=False)
                st.download_button(
                    label="Download Usage Data CSV",
                    data=csv,
                    file_name=f"water_usage_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("👥 Export User Data"):
                user_data = pd.read_sql("SELECT * FROM users", conn)
                csv = user_data.to_csv(index=False)
                st.download_button(
                    label="Download User Data CSV",
                    data=csv,
                    file_name=f"user_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("⚡ Export Power Data"):
                power_data = pd.read_sql("SELECT * FROM power_logs", conn)
                csv = power_data.to_csv(index=False)
                st.download_button(
                    label="Download Power Data CSV",
                    data=csv,
                    file_name=f"power_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        # System configuration
        st.subheader("⚙️ System Configuration")
        
        st.write("**Fetch Cooldown Settings:**")
        current_cooldown = 24  # hours
        new_cooldown = st.slider("Fetch Cooldown (hours)", 1, 72, current_cooldown)
        
        if st.button("Update Cooldown Setting"):
            st.success(f"✅ Fetch cooldown updated to {new_cooldown} hours")
            # In a real system, this would update a configuration table
        
        # Logout button
        if st.button("🚪 Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()
        
        conn.close()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🌊 <strong>Malete Smart Water Management System</strong> 🌊</p>
    <p>Powered by AI • Built with Streamlit & XGBoost</p>
    <p>Ensuring Fair & Sustainable Water Distribution</p>
</div>
""", unsafe_allow_html=True)

# Real-time auto-refresh for dashboard (optional)
if page == "Dashboard":
    import time
    if st.sidebar.checkbox("Auto-refresh Dashboard", value=False):
        time.sleep(30)
        st.rerun()
