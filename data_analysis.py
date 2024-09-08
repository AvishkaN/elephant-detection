import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os

def getDatInsights():

    def generate_synthetic_data(num_entries=500):
        data = []
        start_time = datetime(2024, 5, 5, 16, 30)

        for i in range(num_entries):
            timestamp = start_time + timedelta(seconds=np.random.randint(30, 600))
            status = np.random.choice([True, False], p=[0.3, 0.7])
            
            if status:
                temperature = round(np.random.uniform(25, 35), 2)
                humidity = round(np.random.uniform(60, 100), 2)
                lux_level = round(np.random.uniform(30, 100), 2)
            else:
                temperature = round(np.random.uniform(20, 25), 2)
                humidity = round(np.random.uniform(50, 60), 2)
                lux_level = round(np.random.uniform(0, 30), 2)

            data.append({
                "ID": i + 1,
                "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Status": status,
                "Temperature": temperature,
                "Humidity": humidity,
                "Lux Level": lux_level
            })

            start_time = timestamp
        return pd.DataFrame(data)

    # Create a timestamped folder with hyphens instead of colons
    folder_name = datetime.now().strftime("%H-%M-%S %d-%m-%Y")
    os.makedirs(folder_name, exist_ok=True)

    # Save the full synthetic data to a CSV file inside the folder
    df = generate_synthetic_data(1000)
    df.to_csv(os.path.join(folder_name, '2_all_data.csv'), index=False)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Temperature'] = pd.to_numeric(df['Temperature'])
    df['Humidity'] = pd.to_numeric(df['Humidity'])
    df['Lux Level'] = pd.to_numeric(df['Lux Level'])

    # 1. Detection Frequency
    detection_counts = df['Status'].value_counts()
    detection_counts.index = ["Elephant Leaved", "Elephant Entered"]
    detection_counts.name = 'Detection Frequency'

    # 2. Detection Rate Over Time
    tempDF = df.copy()
    tempDF.set_index('Timestamp', inplace=True)
    hourly_detection_rate = tempDF.resample('h')['Status'].sum()
    hourly_detection_rate.name = 'Hourly Detection Rate'

    # Save the hourly detection rate chart as an image
    plt.figure(figsize=(10, 6))
    hourly_detection_rate.plot(kind='bar', title='Hourly Detection Rate')
    plt.xlabel('Hour')
    plt.ylabel('Number of Detections')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '3_hourly_detection_rate.png'))
    plt.close()

    # 3. Environmental Conditions During Detections
    detected_conditions = df[df['Status'] == True][['Temperature', 'Humidity', 'Lux Level']].mean()
    not_detected_conditions = df[df['Status'] == False][['Temperature', 'Humidity', 'Lux Level']].mean()

    total_detection_count = detection_counts.sum()

    conditions = pd.DataFrame({
        'Detected Conditions': detected_conditions,
        'Not Detected Conditions': not_detected_conditions
    })

    # 4. Duration of Detection Events
    df['Time Difference'] = df['Timestamp'].diff().fillna(pd.Timedelta(seconds=0))
    df['Time Difference (Seconds)'] = df['Time Difference'].dt.total_seconds()

    # 5. Time Between Detections
    inter_detection_time = df[df['Status'] == True]['Timestamp'].diff().dropna()
    inter_detection_time.name = 'Time Between Detections'

    # 6. Trend Analysis
    tempDF['Date'] = tempDF.index.date
    daily_trend = tempDF.groupby('Date')['Status'].sum()
    daily_trend.name = 'Daily Detection Trend'

    # 6. Trend Analysis - Save Daily Detection Trend Chart
    plt.figure(figsize=(12, 6))
    daily_trend.plot(kind='line', marker='o', title='Daily Detection Trend', color='b')
    plt.xlabel('Date')
    plt.ylabel('Number of Detections')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '6_daily_detection_trend.png'))
    plt.close()


    # 7. Correlation Analysis
    df['Status'] = df['Status'].astype(int)
    correlation_matrix = df[['Temperature', 'Humidity', 'Lux Level', 'Status']].corr()

    # Save the correlation matrix heatmap as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '7_correlation_matrix.png'))
    plt.close()

    # Create CSV output
    csv_output = []

    csv_output.append(["1.All Detection Data (please refer 2_all_data.csv file)"])
    csv_output.append([""])

    csv_output.append(["1.Total Detection Count"])
    csv_output.append([total_detection_count])
    csv_output.append([""])

    csv_output.append(["2. Detection Frequency"])
    csv_output.append(detection_counts.reset_index().values.tolist())
    csv_output.append([""])

    csv_output.append(["3. Hourly Detection Rate (please refer 3_hourly_detection_rate.png chart image)"])
    csv_output.append([""])

    csv_output.append(["4. Environmental Conditions During Detections (for more please refer 4_environmental_conditions.png)"])
    csv_output.append([["Condition Type", "Temperature", "Humidity", "Lux Level"]])
    csv_output.append(conditions.T.reset_index().values.tolist())
    csv_output.append([""])

    plt.figure(figsize=(12, 6))
    conditions.plot(kind='bar', title='Environmental Conditions During Detections')
    plt.xlabel('Condition Type')
    plt.ylabel('Average Measurement')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '4_environmental_conditions.png'))
    plt.close()

    df['Time Difference (Minutes)'] = (df['Time Difference'].dt.total_seconds() / 60.0).round(2)
    df[['Timestamp', 'Time Difference (Minutes)']].to_csv(os.path.join(folder_name, '5_detection_events.csv'), index=False)

    csv_output.append(["5. Duration of Detection Events (for more please refer 5_detection_events.csv file)"])
    csv_output.append(["Timestamp , Time Difference (Minutes)"])
    for _, row in df[['Timestamp', 'Time Difference (Minutes)']].head(20).iterrows():
        csv_output.append([[row['Timestamp'], row['Time Difference (Minutes)']]])
    csv_output.append([""])

    csv_output.append(["7. Daily Detection Trend (please refer 6_daily_detection_trend.png)"])
    csv_output.append(daily_trend.reset_index().values.tolist())
    csv_output.append([""])

    csv_output.append(["8. Correlation Matrix (please refer 7_correlation_matrix.png chart image)"])
    csv_output.append(correlation_matrix.reset_index().values.tolist())
    csv_output.append([""])

    final_output_df = pd.DataFrame([item for sublist in csv_output for item in sublist])
    final_output_df.to_csv(os.path.join(folder_name, '1_combined_analysis_with_chart.csv'), index=False, header=False)

