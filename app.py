import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objs as go
import time
from scipy.fft import fft
from fpdf import FPDF
import datetime

# --- PDF Generation Function ---
def generate_pdf_report(damage_detected, metrics):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Bridge Structural Health Monitoring Report", ln=True, align='C')
    pdf.ln(5)
    
    # Date
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    # Model Metrics
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="1. AI Model Verification Metrics", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, txt=f"Accuracy: {metrics['Accuracy']:.2f}%", ln=True)
    pdf.cell(0, 8, txt=f"Precision: {metrics['Precision']:.2f}", ln=True)
    pdf.cell(0, 8, txt=f"Recall: {metrics['Recall']:.2f}", ln=True)
    pdf.cell(0, 8, txt=f"F1-Score: {metrics['F1']:.2f}", ln=True)
    pdf.ln(10)
    
    # Final Verdict
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="2. Live Telemetry Final Assessment", ln=True)
    pdf.set_font("Arial", 'B', 12)
    
    if damage_detected:
        pdf.set_text_color(220, 53, 69)  # Red text for damage
        pdf.cell(0, 10, txt="VERDICT: NOT HEALTHY", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, txt="The AI detected instances of structural anomalies during the telemetry stream. Physical inspection is highly recommended.")
    else:
        pdf.set_text_color(40, 167, 69)  # Green text for healthy
        pdf.cell(0, 10, txt="VERDICT: HEALTHY", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, txt="The structural dynamics remained entirely within normal parameters for the duration of the stream.")
        
    return pdf.output(dest="S").encode("latin1")


# --- UI Styling ---
st.set_page_config(page_title="Bridge SHM System", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #00ff00; font-family: 'Courier New', Courier, monospace; }
    h1, h2, h3 { color: #00ff00; text-shadow: 0 0 5px #00ff00; }
    .stButton>button { color: #0d1117; background-color: #00ff00; border: 1px solid #00ff00; }
    </style>
""", unsafe_allow_html=True)

st.title("🚧 Bridge Structural Health Monitoring [AI-CORE]")
st.markdown("### Production Pipeline: Offline Training & Live Inference")

# --- Feature Extraction ---
def extract_features(data_window):
    features = {}
    features['Mean'] = np.mean(data_window)
    features['Std_Dev'] = np.std(data_window)
    features['RMS'] = np.sqrt(np.mean(data_window**2))
    features['Peak_to_Peak'] = np.max(data_window) - np.min(data_window)
    yf = fft(data_window)
    features['Dominant_Freq_Mag'] = np.max(np.abs(yf[1:len(yf)//2])) 
    return features

# --- Sidebar Configuration ---
st.sidebar.header("🔧 System Configuration")
sim_speed = st.sidebar.slider("Simulation Speed (Seconds)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

# Uploader 1: Training Data
st.sidebar.subheader("1. Offline Training Phase")
training_file = st.sidebar.file_uploader("Upload Labeled Training Data (CSV)", type=["csv"], key="train")

# Uploader 2: Live Inference Data
st.sidebar.subheader("2. Online Inference Phase")
live_file = st.sidebar.file_uploader("Upload Unlabeled Live Telemetry (CSV)", type=["csv"], key="live")

# --- Phase 1: Train the Model ---
if training_file is not None:
    df_train = pd.read_csv(training_file)
    
    if 'Sensor_Reading' in df_train.columns and 'Status' in df_train.columns:
        st.success("Training Data Loaded. Extracting Features...")
        
        window_size = 50
        X_train_list = []
        y_train_list = []
        
        for i in range(0, len(df_train) - window_size, window_size):
            window = df_train['Sensor_Reading'].iloc[i:i+window_size].values
            
            # The Pessimistic Approach: If ANY reading in the window is damaged (1), flag the whole window
            label = 1 if df_train['Status'].iloc[i:i+window_size].sum() > 0 else 0
            
            X_train_list.append(extract_features(window))
            y_train_list.append(label)
            
        X_train_df = pd.DataFrame(X_train_list)
        y_train_array = np.array(y_train_list)
        
        st.subheader("1. Extracted Feature Matrix & Labels")
        display_df = X_train_df.copy()
        display_df['Target_Label'] = y_train_array
        st.dataframe(display_df.head(10), width="stretch") # Shows the first 10 windows
        st.caption("Displaying the first 10 data windows. 'Target_Label' represents the ground truth (0=Healthy, 1=Damaged).")
        
        st.subheader("2. Model Evaluation Metrics")
        
        # Split the uploaded training data to test accuracy before live deployment
        X_fit, X_eval, y_fit, y_eval = train_test_split(X_train_df, y_train_array, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_fit, y_fit)
        
        # Generate predictions on the 30% holdout set
        preds = model.predict(X_eval)
        
        # Save metrics for PDF rendering
        metrics = {
            "Accuracy": accuracy_score(y_eval, preds) * 100,
            "Precision": precision_score(y_eval, preds, zero_division=0),
            "Recall": recall_score(y_eval, preds, zero_division=0),
            "F1": f1_score(y_eval, preds, zero_division=0)
        }
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.2f}%")
        col2.metric("Precision", f"{metrics['Precision']:.2f}")
        col3.metric("Recall", f"{metrics['Recall']:.2f}")
        col4.metric("F1-Score", f"{metrics['F1']:.2f}")
        
        st.info("✅ AI Model successfully trained and verified. Ready for blind inference.")
        st.markdown("---")
        
        # --- Phase 2: Live Inference ---
        if live_file is not None:
            st.header("3. Real-Time Telemetry Inference (Blind Data)")
            df_live = pd.read_csv(live_file)
            
            # Strict Check: Ensure the live data does NOT have the answers!
            if 'Sensor_Reading' in df_live.columns and 'Status' not in df_live.columns:
                
                if st.button("Start Live Monitoring Stream"):
                    chart_placeholder = st.empty()
                    alert_placeholder = st.empty()
                    
                    live_data_y = []
                    live_data_x = []
                    
                    damage_detected = False 
                    
                    for i in range(0, len(df_live) - window_size, window_size):
                        window = df_live['Sensor_Reading'].iloc[i:i+window_size].values
                        current_features = extract_features(window)
                        
                        # INFERENCE: Model makes a prediction completely blindly
                        feature_df = pd.DataFrame([current_features])
                        pred = model.predict(feature_df)[0]
                        
                        live_data_x.append(i)
                        live_data_y.append(current_features['RMS'])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=live_data_x, y=live_data_y, mode='lines', 
                                                 line=dict(color='#00ff00', width=2), name="RMS Vibration"))
                        fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', 
                                          font=dict(color='#00ff00'), margin=dict(l=0, r=0, t=30, b=0),
                                          title="Live Sensor RMS Data")
                        chart_placeholder.plotly_chart(fig, width="stretch")
                        
                        if pred == 1:
                            damage_detected = True 
                            alert_placeholder.error("🚨 CRITICAL ALERT: Structural Anomaly Detected in Live Stream!")
                        else:
                            alert_placeholder.success("✅ Status Nominal: Bridge is healthy.")
                        
                        time.sleep(sim_speed)
                        
                        if len(live_data_x) > 20:
                            live_data_x.pop(0)
                            live_data_y.pop(0)
                            
                    # --- Final Assessment & PDF Download ---
                    st.markdown("---")
                    st.subheader("🏁 Final Run Assessment")
                    
                    if damage_detected:
                        st.error("❌ FINAL VERDICT: NOT HEALTHY. The AI detected instances of structural anomalies during the telemetry stream. Physical inspection is highly recommended.")
                    else:
                        st.success("✅ FINAL VERDICT: HEALTHY. The structural dynamics remained entirely within normal parameters for the duration of the stream.")
                        
                    # Generate and render PDF Download Button
                    pdf_bytes = generate_pdf_report(damage_detected, metrics)
                    st.download_button(
                        label="📄 Download Final PDF Report",
                        data=pdf_bytes,
                        file_name="Bridge_SHM_Report.pdf",
                        mime="application/pdf"
                    )
                        
            else:
                st.error("Live telemetry must contain ONLY 'Sensor_Reading'. Please remove the 'Status' column to prove the AI works on blind data.")
        else:
            st.warning("Awaiting live telemetry stream. Please upload the unlabeled 'live_telemetry.csv' in Step 2.")
    else:
        st.error("Training dataset must contain both 'Sensor_Reading' and 'Status' columns.")
else:
    st.info("Step 1: Upload historical training data (bridge_data_v2.csv) to teach the AI.")
