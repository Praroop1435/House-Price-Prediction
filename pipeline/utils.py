# utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_market_comparison(df, city_encoded, size_in_sqft, predicted_price, selected_city):
    """
    Plots the market comparison between city average price and predicted price.
    """
    # Compute city average price per sqft
    city_avg_per_sqft = df[df["City_encoded"] == city_encoded]["Price_per_SqFt"].mean()

    if not np.isnan(city_avg_per_sqft):
        city_avg_price = city_avg_per_sqft * size_in_sqft

        fig, ax = plt.subplots(figsize=(8, 4))

        # City average bar
        ax.barh([selected_city], [city_avg_price],
                color="skyblue", alpha=0.7, label="Avg City Price")

        # Predicted price line
        ax.axvline(predicted_price, color="red", linestyle="--", linewidth=2, label="Predicted Price")

        ax.set_xlabel("Price (Lakhs)")
        ax.set_title("Market Comparison")
        ax.legend()

        return fig
    else:
        return None
