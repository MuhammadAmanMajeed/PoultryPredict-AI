import pandas as pd
import numpy as np

def generate_poultry_data(num_records=15000):
    np.random.seed(42)
    
    # Base flocks
    flock_sizes = np.random.randint(1000, 50000, num_records)
    
    # Environment
    temp = np.random.normal(24, 4, num_records) # Mean 24C, std 4
    humidity = np.random.normal(60, 10, num_records) # Mean 60%, std 10
    ammonia = np.random.exponential(8, num_records) # Mostly low, some spikes
    light_lux = np.random.normal(40, 10, num_records)
    light_duration = np.random.normal(15, 1, num_records)
    noise = np.random.normal(45, 10, num_records)
    
    # Feed (approx 100g to 130g per bird)
    feed_per_bird = np.random.normal(0.115, 0.010, num_records)
    feed_total_kg = flock_sizes * feed_per_bird
    
    # Egg Production Calculation (Base efficiency ~85% for commercial layer)
    base_eff = 0.88
    
    # Biological Penalties
    temp_penalty = np.where((temp > 28) | (temp < 18), abs(temp - 24) * 0.008, 0)
    hum_penalty = np.where((humidity > 75) | (humidity < 40), abs(humidity - 60) * 0.004, 0)
    amm_penalty = np.where(ammonia > 15, ammonia * 0.002, 0)
    light_penalty = np.where(light_duration < 14, (14 - light_duration) * 0.01, 0)
    
    final_eff = base_eff - temp_penalty - hum_penalty - amm_penalty - light_penalty
    # Add noise for realistic model training variance
    final_eff = np.clip(final_eff + np.random.normal(0, 0.03, num_records), 0.35, 0.98)
    
    eggs_produced = (flock_sizes * final_eff).astype(int)
    
    # Generate random dates for the dataset
    dates = pd.date_range(start="2024-01-01", periods=num_records, freq="H").strftime("%m/%d/%Y").tolist()
    
    # Generate random hour ranges
    hours = [f"{str(h).zfill(2)}:00-{str(h+1).zfill(2)}:00" for h in np.random.randint(0, 23, num_records)]

    df = pd.DataFrame({
        'Date': dates,
        'Hour_Range': hours,
        'Amount_of_chicken': flock_sizes,
        'Amount_of_Feeding': feed_total_kg,
        'Ammonia': np.round(ammonia, 2),
        'Temperature': np.round(temp, 2),
        'Humidity': np.round(humidity, 2),
        'Light_Intensity': np.round(light_lux, 2),
        'Light_Duration': np.round(light_duration, 2),
        'Noise': np.round(noise, 2),
        'Total_egg_production': eggs_produced
    })
    
    # Clean bounds
    df['Ammonia'] = df['Ammonia'].clip(0, 100)
    df['Humidity'] = df['Humidity'].clip(0, 100)
    df['Light_Intensity'] = df['Light_Intensity'].clip(0, 100)
    
    # Save massive dataset
    df.to_csv("Egg_Production_Large.csv", index=False)
    print(f"Successfully generated {num_records} rows of advanced synthetic data in Egg_Production_Large.csv")

if __name__ == "__main__":
    generate_poultry_data()
